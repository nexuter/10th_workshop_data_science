from __future__ import annotations

import math

import numpy as np


def as_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(mask, dtype=float) >= threshold).astype(np.uint8)


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def wind_to_offset(direction_deg: float, distance: int = 1) -> tuple[int, int]:
    """Convert meteorological wind-from degrees to row/col downwind offset."""
    to_deg = (direction_deg + 180.0) % 360.0
    radians = math.radians(to_deg)
    row = int(round(-math.cos(radians) * distance))
    col = int(round(math.sin(radians) * distance))
    return row, col


def shift(mask: np.ndarray, row_offset: int, col_offset: int) -> np.ndarray:
    arr = np.asarray(mask)
    out = np.zeros_like(arr)
    rows, cols = arr.shape

    src_r0 = max(0, -row_offset)
    src_r1 = min(rows, rows - row_offset)
    dst_r0 = max(0, row_offset)
    dst_r1 = min(rows, rows + row_offset)

    src_c0 = max(0, -col_offset)
    src_c1 = min(cols, cols - col_offset)
    dst_c0 = max(0, col_offset)
    dst_c1 = min(cols, cols + col_offset)

    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    base = as_binary(mask)
    out = base.copy()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                out = np.maximum(out, shift(base, dr, dc))
    return out.astype(np.uint8)


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    base = as_binary(mask)
    out = np.ones_like(base, dtype=np.uint8)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                out = np.minimum(out, shift(base, dr, dc))
    return out.astype(np.uint8)


def boundary(mask: np.ndarray) -> np.ndarray:
    base = as_binary(mask)
    return np.maximum(dilate(base, 1) - erode(base, 1), 0).astype(np.uint8)


def burn_front(burn_map: np.ndarray) -> np.ndarray:
    burned = as_binary(burn_map)
    return np.maximum(dilate(burned, 1) - burned, 0).astype(np.uint8)


def bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(np.asarray(mask) > 0)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    rows, cols = mask.shape
    return max(0, r0 - pad), max(0, c0 - pad), min(rows, r1 + pad), min(cols, c1 + pad)
