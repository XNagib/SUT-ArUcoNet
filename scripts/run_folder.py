#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sut_aruconet import SUTArUcoNet
def main()->int:
    p=argparse.ArgumentParser(description='Run SUT-ArUcoNet on every image in a folder.'); p.add_argument('folder',type=Path); p.add_argument('--model-dir',type=Path,default=Path('models/SUT-ArUcoNet-v1')); p.add_argument('--output',type=Path,default=Path('outputs/folder_results.json')); p.add_argument('--imgsz',type=int); p.add_argument('--conf',type=float); p.add_argument('--max-hamming',type=int); p.add_argument('--device',default='auto'); a=p.parse_args(); model=SUTArUcoNet(a.model_dir,device=a.device); images=sorted([x for x in a.folder.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]); results=[model.detect_path(x,imgsz=a.imgsz,conf=a.conf,max_hamming=a.max_hamming) for x in images]; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'images':len(results),'results':results},indent=2),encoding='utf-8'); print(f'wrote {a.output}'); return 0
if __name__=='__main__': raise SystemExit(main())
