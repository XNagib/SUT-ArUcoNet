#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
def main()->int:
    p=argparse.ArgumentParser(description='Export detector to ONNX.'); p.add_argument('--model-dir',type=Path,default=Path('models/SUT-ArUcoNet-v1')); p.add_argument('--imgsz',type=int,default=416); a=p.parse_args(); YOLO(str(a.model_dir/'detector.pt')).export(format='onnx',imgsz=a.imgsz); return 0
if __name__=='__main__': raise SystemExit(main())
