from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cv_practice.features.edges import canny
from cv_practice.io.image_io import to_gray


@dataclass(frozen=True)
class BaselineConfig:
    blur_ksize: int = 5
    canny_low: int = 100
    canny_high: int = 200


def run_baseline(
    bgr: np.ndarray, cfg: BaselineConfig | None = None
) -> dict[str, np.ndarray]:
    if cfg is None:
        cfg = BaselineConfig()

    gray = to_gray(bgr)
    if cfg.blur_ksize % 2 == 0:
        raise ValueError("blur_ksize must be odd for GaussianBlur.")
    gray_blur = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    edges = canny(gray_blur, low=cfg.canny_low, high=cfg.canny_high)
    return {"gray": gray, "gray_blur": gray_blur, "edges": edges}