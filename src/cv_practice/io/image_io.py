from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray:
    """Read an image in BGR (OpenCV default). Raises if not found/unreadable."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to grayscale uint8."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)