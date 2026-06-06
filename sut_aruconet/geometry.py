"""Geometry helpers for SUT-ArUcoNet inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropTransform:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(1e-6, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1e-6, self.y2 - self.y1)


def expand_xyxy(xyxy: np.ndarray, image_width: int, image_height: int, padding: float = 0.20, square: bool = True) -> CropTransform:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if square:
        side = max(width, height) * (1.0 + 2.0 * padding)
        width = side
        height = side
    else:
        width *= 1.0 + 2.0 * padding
        height *= 1.0 + 2.0 * padding
    return CropTransform(max(0.0, cx - width / 2.0), max(0.0, cy - height / 2.0), min(float(image_width - 1), cx + width / 2.0), min(float(image_height - 1), cy + height / 2.0))


def crop_and_resize(image: np.ndarray, crop: CropTransform, size: int = 64) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in (crop.x1, crop.y1, crop.x2, crop.y2)]
    x1 = max(0, min(image.shape[1] - 1, x1))
    x2 = max(x1 + 1, min(image.shape[1], x2))
    y1 = max(0, min(image.shape[0] - 1, y1))
    y2 = max(y1 + 1, min(image.shape[0], y2))
    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


def corners_from_crop(corners: np.ndarray, crop: CropTransform) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = crop.x1 + pts[:, 0] * crop.width
    pts[:, 1] = crop.y1 + pts[:, 1] * crop.height
    return pts


def order_corners_geometric(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def logical_from_geometric(geometric_corners: np.ndarray, rotation_to_canonical: int) -> np.ndarray:
    pts = np.asarray(geometric_corners, dtype=np.float32).reshape(4, 2)
    indices = [(idx + rotation_to_canonical) % 4 for idx in range(4)]
    return pts[indices]


def rectify_marker(image: np.ndarray, geometric_corners: np.ndarray, size: int = 64) -> np.ndarray:
    src = np.asarray(geometric_corners, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0.0, 0.0], [size - 1.0, 0.0], [size - 1.0, size - 1.0], [0.0, size - 1.0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (size, size), flags=cv2.INTER_LINEAR)


def solve_marker_pnp(logical_corners: np.ndarray, marker_size: float, camera_matrix: np.ndarray, distortion: np.ndarray | None = None):
    image_points = np.asarray(logical_corners, dtype=np.float32).reshape(4, 2)
    half = float(marker_size) / 2.0
    object_points = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32) if distortion is None else distortion.astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix.astype(np.float32), dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
    error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)))
    return rvec.reshape(3), tvec.reshape(3), error
