import numpy as np

from cv_practice.features.edges import canny


def test_canny_output_shape():
    gray = np.zeros((64, 64), dtype=np.uint8)
    edges = canny(gray)
    assert edges.shape == gray.shape
    assert edges.dtype == np.uint8