#!/usr/bin/env python3
from __future__ import annotations
import argparse,time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2,numpy as np
from sut_aruconet import SUTArUcoNet
def main()->int:
    p=argparse.ArgumentParser(description='Run SUT-ArUcoNet on a live camera.'); p.add_argument('--camera',type=int,default=0); p.add_argument('--model-dir',type=Path,default=Path('models/SUT-ArUcoNet-v1')); p.add_argument('--width',type=int,default=640); p.add_argument('--height',type=int,default=480); p.add_argument('--camera-fps',type=float,default=0.0); p.add_argument('--process-every',type=int,default=1); p.add_argument('--imgsz',type=int); p.add_argument('--conf',type=float); p.add_argument('--max-hamming',type=int); p.add_argument('--device',default='auto'); a=p.parse_args(); model=SUTArUcoNet(a.model_dir,device=a.device); cap=cv2.VideoCapture(a.camera,cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_FRAME_WIDTH,a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,a.height);
    if a.camera_fps>0: cap.set(cv2.CAP_PROP_FPS,a.camera_fps)
    if not cap.isOpened(): raise SystemExit(f'Could not open camera {a.camera}')
    cv2.namedWindow('SUT-ArUcoNet',cv2.WINDOW_NORMAL); frame_index=0; last={'markers':[],'detections':0,'runtime_ms':{'wall':0.0}}
    while True:
        ok,frame=cap.read();
        if not ok: continue
        frame_index+=1
        if frame_index%max(1,a.process_every)==0:
            t0=time.perf_counter(); last=model.detect_image(frame,imgsz=a.imgsz,conf=a.conf,max_hamming=a.max_hamming); last['runtime_ms']['wall']=(time.perf_counter()-t0)*1000.0
        overlay=frame.copy()
        for marker in last['markers']:
            pts=np.asarray(marker['corners'],dtype=np.int32); color=(0,255,0) if marker.get('decoded') else (0,255,255); cv2.polylines(overlay,[pts],True,color,2,cv2.LINE_AA); cv2.putText(overlay,str(marker.get('id')),tuple(np.mean(pts,axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        cv2.putText(overlay,f"det {last.get('detections',0)} markers {len(last.get('markers',[]))} {last.get('runtime_ms',{}).get('wall',0.0):.1f} ms",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2); cv2.imshow('SUT-ArUcoNet',overlay); key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),27): break
    cap.release(); cv2.destroyAllWindows(); return 0
if __name__=='__main__': raise SystemExit(main())
