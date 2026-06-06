"""Deterministic 4x4_1000 marker decoder for SUT-ArUcoNet."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    marker_id: int
    rotation_to_canonical: int
    hamming_distance: int
    second_best_distance: int
    margin: float
    confidence: float
    bits: np.ndarray


def decode_rectified_4x4_1000(rectified: np.ndarray) -> DecodeResult | None:
    bits = sample_aruco_4x4_bits(rectified)
    return decode_aruco_4x4_1000_bits(bits)


def decode_aruco_4x4_1000_bits(bits: np.ndarray) -> DecodeResult | None:
    bits = np.asarray(bits, dtype=np.uint8).reshape(4, 4)
    dictionary_bits = _dict_4x4_1000_bits()
    best = None
    second_best = 10**9
    for rotation in range(4):
        candidate = np.rot90(bits, rotation)
        distances = np.count_nonzero(dictionary_bits != candidate[None, :, :], axis=(1, 2))
        marker_id = int(np.argmin(distances))
        distance = int(distances[marker_id])
        if best is None or distance < best[2]:
            if best is not None:
                second_best = min(second_best, best[2])
            best = (marker_id, rotation, distance)
        else:
            second_best = min(second_best, distance)
    if best is None:
        return None
    marker_id, rotation, distance = best
    second_best = second_best if second_best != 10**9 else int(bits.size)
    max_bits = float(bits.size)
    margin = max(0.0, min(1.0, (second_best - distance) / max_bits))
    confidence = max(0.0, min(1.0, 1.0 - distance / max_bits))
    return DecodeResult(marker_id, rotation, distance, int(second_best), margin, confidence, bits)


def sample_aruco_4x4_bits(rectified: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
    grid_cells = 6
    normalized = cv2.resize(gray, (grid_cells * 10, grid_cells * 10), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cell = binary.shape[0] / float(grid_cells)
    bits = np.zeros((4, 4), dtype=np.uint8)
    for row in range(4):
        for col in range(4):
            cx = int(round((col + 1.5) * cell))
            cy = int(round((row + 1.5) * cell))
            x1 = max(0, int(round(cx - cell * 0.18)))
            x2 = min(binary.shape[1], int(round(cx + cell * 0.18)))
            y1 = max(0, int(round(cy - cell * 0.18)))
            y2 = min(binary.shape[0], int(round(cy + cell * 0.18)))
            bits[row, col] = 1 if float(np.mean(binary[y1:y2, x1:x2])) > 127.0 else 0
    return bits


@lru_cache(maxsize=1)
def _dict_4x4_1000_bits() -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    bits = np.zeros((1000, 4, 4), dtype=np.uint8)
    for marker_id in range(1000):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 60)
        bits[marker_id] = sample_aruco_4x4_bits(marker)
    return bits
