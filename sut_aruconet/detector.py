"""Public SUT-ArUcoNet stable inference pipeline."""

from __future__ import annotations

import json, time
from pathlib import Path
from typing import Any

import cv2, numpy as np, torch
from torch import nn
from ultralytics import YOLO

from .decoder import decode_rectified_4x4_1000
from .geometry import corners_from_crop, crop_and_resize, expand_xyxy, logical_from_geometric, order_corners_geometric, rectify_marker, solve_marker_pnp

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int=1)->None:
        super().__init__(); self.block=nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride=stride,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.SiLU(inplace=True))
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.block(x)

class CornerRefiner64(nn.Module):
    def __init__(self)->None:
        super().__init__(); self.features=nn.Sequential(ConvBlock(3,16,2),ConvBlock(16,24),ConvBlock(24,32,2),ConvBlock(32,48),ConvBlock(48,64,2),ConvBlock(64,96),ConvBlock(96,128,2),nn.AdaptiveAvgPool2d(1)); self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,96),nn.SiLU(inplace=True),nn.Dropout(0.05),nn.Linear(96,8),nn.Sigmoid())
    def forward(self,x:torch.Tensor)->torch.Tensor: return self.head(self.features(x)).view(-1,4,2)

class SUTArUcoNet:
    def __init__(self, model_dir:str|Path|None=None, device:str='auto')->None:
        self.model_dir=Path(model_dir) if model_dir else Path(__file__).resolve().parents[1]/'models'/'SUT-ArUcoNet-v1'
        self.manifest=json.loads((self.model_dir/'model_manifest.json').read_text(encoding='utf-8')); self.defaults=dict(self.manifest.get('runtime_defaults',{}))
        self.torch_device,self.yolo_device=self._resolve_device(device); self.detector=YOLO(str(self.model_dir/self.manifest['assets']['detector']['file'])); self.corner_refiner=self._load_corner_refiner(self.model_dir/self.manifest['assets']['corner_refiner']['file'])
    def detect_path(self,image_path:str|Path,**kwargs:Any)->dict[str,Any]:
        image_path=Path(image_path); image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f'Could not read image: {image_path}')
        result=self.detect_image(image,**kwargs); result['image']=str(image_path); return result
    def detect_image(self,image:np.ndarray,imgsz:int|None=None,conf:float|None=None,iou:float|None=None,max_hamming:int|None=None,padding:float|None=None,marker_size:float=0.0,camera_matrix:np.ndarray|None=None,distortion:np.ndarray|None=None)->dict[str,Any]:
        imgsz=int(imgsz if imgsz is not None else self.defaults.get('imgsz',416)); conf=float(conf if conf is not None else self.defaults.get('conf',0.25)); iou=float(iou if iou is not None else self.defaults.get('iou',0.5)); max_hamming=int(max_hamming if max_hamming is not None else self.defaults.get('max_hamming',0)); padding=float(padding if padding is not None else self.defaults.get('padding',0.2))
        t0=time.perf_counter(); prediction=self.detector.predict(source=image,imgsz=imgsz,conf=conf,iou=iou,device=self.yolo_device,verbose=False)[0]; detections=self._boxes_from_prediction(prediction); detector_ms=(time.perf_counter()-t0)*1000.0
        t1=time.perf_counter(); markers=self._process_detections(image,detections,padding,max_hamming,marker_size,camera_matrix,distortion); post_ms=(time.perf_counter()-t1)*1000.0
        return {'model':self.manifest['name'],'version':self.manifest['version'],'dictionary':'4x4_1000','image_size':{'width':int(image.shape[1]),'height':int(image.shape[0])},'settings':{'imgsz':imgsz,'conf':conf,'iou':iou,'max_hamming':max_hamming,'padding':padding},'detections':len(detections),'markers':markers,'runtime_ms':{'detector':detector_ms,'postprocess':post_ms,'total':detector_ms+post_ms}}
    def _resolve_device(self,device:str)->tuple[torch.device,str]:
        if device in ('auto','0'): return (torch.device('cuda:0'),'0') if torch.cuda.is_available() else (torch.device('cpu'),'cpu')
        return torch.device(device),device
    def _load_corner_refiner(self,path:Path)->CornerRefiner64:
        checkpoint=torch.load(path,map_location=self.torch_device); model=CornerRefiner64().to(self.torch_device); model.load_state_dict(checkpoint['model_state']); model.eval(); return model
    @staticmethod
    def _boxes_from_prediction(prediction:Any)->list[dict[str,Any]]:
        boxes=prediction.boxes
        if boxes is None or boxes.xyxy is None: return []
        xyxy=boxes.xyxy.detach().cpu().numpy(); scores=boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        return [{'xyxy':np.asarray(box,dtype=np.float32),'confidence':float(score)} for box,score in zip(xyxy,scores)]
    def _process_detections(self,image,detections,padding,max_hamming,marker_size,camera_matrix,distortion):
        if not detections: return []
        crops=[expand_xyxy(d['xyxy'],image.shape[1],image.shape[0],padding=padding,square=True) for d in detections]; crop_images=[crop_and_resize(image,crop,64) for crop in crops]; crop_corners_batch=self._predict_corners_batch(crop_images); geometric_batch=[order_corners_geometric(corners_from_crop(cc,crop)) for cc,crop in zip(crop_corners_batch,crops)]
        markers=[]
        for detection,geometric,crop_corners in zip(detections,geometric_batch,crop_corners_batch):
            decoded=decode_rectified_4x4_1000(rectify_marker(image,geometric,64)); decoded_ok=decoded is not None and decoded.hamming_distance<=max_hamming; corner_conf=float(np.mean(_corner_confidences(crop_corners)))
            if not decoded_ok:
                markers.append({'id':None,'decoded':False,'decode_status':'decode_failed','candidate_id':int(decoded.marker_id) if decoded is not None else None,'detection_confidence':float(detection['confidence']),'corner_confidence':corner_conf,'decoder_confidence':float(decoded.confidence) if decoded is not None else 0.0,'hamming_distance':int(decoded.hamming_distance) if decoded is not None else None,'detection_box_xyxy':np.asarray(detection['xyxy'],dtype=np.float32).tolist(),'geometric_corners':geometric.tolist(),'corners':geometric.tolist()}); continue
            logical=logical_from_geometric(geometric,decoded.rotation_to_canonical); pose=None; reprojection_error=None
            if marker_size>0.0 and camera_matrix is not None:
                pose_result=solve_marker_pnp(logical,marker_size,camera_matrix,distortion)
                if pose_result is not None:
                    rvec,tvec,reprojection_error=pose_result; pose={'rvec':rvec.tolist(),'tvec':tvec.tolist()}
            decoder_conf=float(decoded.confidence); confidence=float(np.clip(0.45*float(detection['confidence'])+0.25*corner_conf+0.30*decoder_conf,0.0,1.0))
            markers.append({'id':int(decoded.marker_id),'decoded':True,'decode_status':'decoded','dictionary':'4x4_1000','confidence':confidence,'detection_confidence':float(detection['confidence']),'corner_confidence':corner_conf,'decoder_confidence':decoder_conf,'decoder_margin':float(decoded.margin),'hamming_distance':int(decoded.hamming_distance),'rotation_to_canonical':int(decoded.rotation_to_canonical),'detection_box_xyxy':np.asarray(detection['xyxy'],dtype=np.float32).tolist(),'corners':logical.tolist(),'geometric_corners':geometric.tolist(),'pose':pose,'reprojection_error_px':reprojection_error})
        return markers
    @torch.inference_mode()
    def _predict_corners_batch(self,crop_images:list[np.ndarray])->np.ndarray:
        if not crop_images: return np.zeros((0,4,2),dtype=np.float32)
        batch=np.stack([cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 for crop in crop_images],axis=0); tensor=torch.from_numpy(batch.transpose(0,3,1,2)).to(self.torch_device); return np.clip(self.corner_refiner(tensor).detach().cpu().numpy(),0.0,1.0)

def _corner_confidences(crop_corners:np.ndarray)->list[float]:
    vals=[]
    for x,y in np.asarray(crop_corners,dtype=np.float32).reshape(4,2):
        margin=min(float(x),float(y),1.0-float(x),1.0-float(y)); vals.append(max(0.0,min(1.0,margin/0.08)))
    return vals
