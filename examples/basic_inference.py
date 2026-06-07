from __future__ import annotations

import json
from pathlib import Path

import cv2

from sut_aruconet import SUTArUcoNet


def run(image_path: str | Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    model = SUTArUcoNet("models/SUT-ArUcoNet-v1")
    return model.detect_image(image, dictionary="4x4_1000")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal SUT-ArUcoNet v1 image inference example.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.image), indent=2))