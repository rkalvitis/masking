"""Pre-enhancement: push the (near-)white background to pure white and stretch contrast.

1. Illumination / white map: per channel, a large-scale grey closing + blur at 1/8 res gives
   the local background brightness (the beetle is filled in by surrounding wall). Dividing by
   it flattens vignetting and wall shading so background -> ~1.0 everywhere.
2. Levels stretch: values >= `hi` of the local white become pure white, values <= `lo` become
   black, in between are stretched linearly (per channel, which also boosts saturation).
"""
from __future__ import annotations
import numpy as np
import cv2


def white_map(rgb: np.ndarray, down: int = 8, kernel_frac: float = 0.12) -> np.ndarray:
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w // down, h // down), interpolation=cv2.INTER_AREA)
    k = int(max(small.shape[:2]) * kernel_frac) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)     # fills dark objects with surrounding wall
    closed = cv2.GaussianBlur(closed, (0, 0), k / 4)
    wm = cv2.resize(closed, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.maximum(wm, 40.0)


def enhance(rgb: np.ndarray, lo: float = 0.02, hi: float = 0.92, gamma: float = 1.0) -> np.ndarray:
    wm = white_map(rgb)
    norm = rgb.astype(np.float32) / wm                         # background ~ 1.0
    out = np.clip((norm - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        out = out ** gamma
    return (out * 255.0 + 0.5).astype(np.uint8)
