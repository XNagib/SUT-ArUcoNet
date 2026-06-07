# Usage

SUT-ArUcoNet v1 is used as a Python package.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Image Array Inference

```python
import cv2
from sut_aruconet import SUTArUcoNet

image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
model = SUTArUcoNet("models/SUT-ArUcoNet-v1")
result = model.detect_image(image, dictionary="4x4_1000")
```

## Settings

Default settings are stored in `models/SUT-ArUcoNet-v1/model_manifest.json`:

- detector input size: `416`
- detector confidence: `0.25`
- NMS IoU: `0.5`
- crop padding: `0.2`
- maximum Hamming distance: `0`
- default dictionary: `4x4_1000`

## Optional MIP Extension

```python
result = model.detect_image(image, dictionary="mip36h12")
```

This path is the **SUT-ArUcoNet v1 MIP extension**. Keep it separate from core `DICT_4X4_1000` use.

## Robot Integration Note

The release intentionally provides importable inference code only. A robot project can wrap `SUTArUcoNet.detect_image()` in a ROS2 node and supply camera frames, calibration, and marker-size configuration in that project.