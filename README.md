# SUT-ArUcoNet v1

SUT-ArUcoNet v1 is a public inference package for ArUco marker detection and decoding. The core release detects `DICT_4X4_1000` markers using a lightweight neural detector, a 64x64 corner refiner, and a deterministic rectified-bit decoder.

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

## Python Usage

```python
import cv2
from sut_aruconet import SUTArUcoNet

image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
model = SUTArUcoNet("models/SUT-ArUcoNet-v1")
result = model.detect_image(image, dictionary="4x4_1000")
```

Each decoded marker includes `id`, `decoded`, `decode_status`, `corners`, `detection_confidence`, `corner_confidence`, `decoder_confidence`, `hamming_distance`, and related geometry fields.

## Dictionaries

Core support: OpenCV `DICT_4X4_1000` through `dictionary="4x4_1000"`.

Optional extension: `dictionary="mip36h12"`.

## Documentation

See:

- `docs/usage.md`
- `docs/model_card.md`
- `docs/architecture.md`

## License

This release is distributed under AGPL-3.0. The detector runtime uses Ultralytics, which is AGPL-3.0 unless a separate commercial license is obtained from its vendor.

## Citation

Citation placeholder: thesis, Object Recognition in ROS Using Neural Networks, Suez University of Technology.