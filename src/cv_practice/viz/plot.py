from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_gray(gray: np.ndarray, title: str = "image") -> None:
    plt.figure()
    plt.title(title)
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.show()


def save_gray(gray: np.ndarray, path: str | Path) -> None:
    """Save grayscale uint8 to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), gray)
    if not ok:
        raise RuntimeError(f"Failed to write image to {path}")