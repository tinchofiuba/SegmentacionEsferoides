import cv2
import numpy as np
import pytest


@pytest.fixture
def make_mask_png(tmp_path):
    """Factory: escribe una máscara taggeada a color (BGR) y devuelve su path.

    `class_regions` es una lista de (color_bgr, (y0, y1, x0, x1)) a pintar
    sobre un fondo negro, imitando el esquema de colores real de
    u_resnet_segesferoides.dataset.MASK_COLOR_TO_CLASS.
    """

    def factory(name, shape=(64, 64), class_regions=()):
        mask = np.zeros((*shape, 3), dtype=np.uint8)
        for color_bgr, (y0, y1, x0, x1) in class_regions:
            mask[y0:y1, x0:x1] = color_bgr
        path = tmp_path / name
        cv2.imwrite(str(path), mask)
        return str(path)

    return factory
