import numpy as np

from cv_practice.io.image_io import to_gray


def test_to_gray_shape_and_dtype():
    bgr = np.zeros((64, 32, 3), dtype=np.uint8)
    gray = to_gray(bgr)
    assert gray.shape == (64, 32)
    assert gray.dtype == np.uint8