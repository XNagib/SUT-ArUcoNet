# Usage

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Image Inference

```powershell
python scripts\run_image.py image.jpg --output outputs\image.json --overlay outputs\image_overlay.jpg
```

## Folder Inference

```powershell
python scripts\run_folder.py images --output outputs\folder_results.json
```

## Camera Inference

```powershell
python scripts\run_camera.py --camera 0 --imgsz 416 --conf 0.25 --process-every 3
```

## Troubleshooting

If CUDA is not used, check `torch.cuda.is_available()`. If no markers are detected, lower `--conf` carefully and inspect lighting, glare, focus, marker size, and marker visibility. If IDs are missing on partial markers, move the camera to reveal the full marker.
