# SUT-ArUcoNet

SUT-ArUcoNet is a public inference release for detecting `DICT_4X4_1000` ArUco markers with a lightweight neural detector, a 64x64 corner refiner, and a deterministic rectified-bit decoder.

This repository contains only the stable inference package and model weights. Datasets, training files, and development material are not included.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## One Image

```powershell
python scripts\run_image.py path\to\image.jpg --output outputs\result.json --overlay outputs\overlay.jpg
```

## Folder

```powershell
python scripts\run_folder.py path\to\images --output outputs\folder_results.json
```

## Live Camera

```powershell
python scripts\run_camera.py --camera 0 --imgsz 416 --conf 0.25 --process-every 3
```

## Output

Each marker record includes `id`, `confidence`, `detection_confidence`, `corner_confidence`, `decoder_confidence`, `hamming_distance`, `corners`, and optional pose fields when camera calibration and marker size are supplied through the Python API.

## License

This release is distributed under AGPL-3.0. The detector runtime uses Ultralytics, which is AGPL-3.0 unless a separate commercial license is obtained from its vendor.

## Citation

Citation placeholder: thesis, Object Recognition in ROS Using Neural Networks, Suez University of Technology.
