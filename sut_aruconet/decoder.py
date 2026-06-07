"""Deterministic marker decoders for SUT-ArUcoNet v1."""

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


def decode_rectified(rectified: np.ndarray, dictionary: str = "4x4_1000") -> DecodeResult | None:
    if dictionary == "4x4_1000":
        return decode_rectified_4x4_1000(rectified)
    if dictionary == "mip36h12":
        return decode_rectified_mip36h12(rectified)
    raise ValueError(f"Unsupported dictionary: {dictionary}")


def decode_rectified_4x4_1000(rectified: np.ndarray) -> DecodeResult | None:
    bits = sample_payload_bits(rectified, payload_cells=4)
    return decode_payload_bits(bits, _dict_4x4_1000_bits())


def decode_rectified_mip36h12(rectified: np.ndarray) -> DecodeResult | None:
    bits = sample_payload_bits(rectified, payload_cells=6)
    return decode_payload_bits(bits, _dict_mip36h12_bits())


def decode_payload_bits(bits: np.ndarray, dictionary_bits: np.ndarray) -> DecodeResult | None:
    bits = np.asarray(bits, dtype=np.uint8)
    best: tuple[int, int, int] | None = None
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


def sample_payload_bits(rectified: np.ndarray, payload_cells: int) -> np.ndarray:
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
    grid_cells = int(payload_cells) + 2
    normalized = cv2.resize(gray, (grid_cells * 10, grid_cells * 10), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cell = binary.shape[0] / float(grid_cells)
    bits = np.zeros((payload_cells, payload_cells), dtype=np.uint8)
    for row in range(payload_cells):
        for col in range(payload_cells):
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
    return _dictionary_bits(cv2.aruco.DICT_4X4_1000, 1000, 4)


@lru_cache(maxsize=1)
def _dict_mip36h12_bits() -> np.ndarray:
    dictionary_id = getattr(cv2.aruco, "DICT_ARUCO_MIP_36H12", None)
    if dictionary_id is None:
        raise RuntimeError("This OpenCV build does not provide DICT_ARUCO_MIP_36H12.")
    return _dictionary_bits(dictionary_id, 250, 6)


def _dictionary_bits(dictionary_id: int, marker_count: int, payload_cells: int) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    bits = np.zeros((marker_count, payload_cells, payload_cells), dtype=np.uint8)
    for marker_id in range(marker_count):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 80)
        bits[marker_id] = sample_payload_bits(marker, payload_cells=payload_cells)
    return bits
