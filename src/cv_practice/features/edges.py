from __future__ import annotations

import cv2
import numpy as np


def canny(gray: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Canny edge detector. Expects grayscale uint8."""
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image with shape (H, W).")
    if gray.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale image.")
    return cv2.Canny(gray, threshold1=low, threshold2=high)