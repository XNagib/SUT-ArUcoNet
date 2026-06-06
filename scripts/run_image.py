#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2,numpy as np
from sut_aruconet import SUTArUcoNet
def main()->int:
    p=argparse.ArgumentParser(description='Run SUT-ArUcoNet on one image.'); p.add_argument('image',type=Path); p.add_argument('--model-dir',type=Path,default=Path('models/SUT-ArUcoNet-v1')); p.add_argument('--output',type=Path); p.add_argument('--overlay',type=Path); p.add_argument('--imgsz',type=int); p.add_argument('--conf',type=float); p.add_argument('--max-hamming',type=int); p.add_argument('--device',default='auto'); a=p.parse_args(); model=SUTArUcoNet(a.model_dir,device=a.device); result=model.detect_path(a.image,imgsz=a.imgsz,conf=a.conf,max_hamming=a.max_hamming); text=json.dumps(result,indent=2); print(text)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text,encoding='utf-8')
    if a.overlay:
        image=cv2.imread(str(a.image),cv2.IMREAD_COLOR)
        for marker in result['markers']:
            pts=np.asarray(marker['corners'],dtype=np.int32); color=(0,255,0) if marker.get('decoded') else (0,255,255); cv2.polylines(image,[pts],True,color,2,cv2.LINE_AA); cv2.putText(image,str(marker.get('id')),tuple(np.mean(pts,axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        a.overlay.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(a.overlay),image)
    return 0
if __name__=='__main__': raise SystemExit(main())
