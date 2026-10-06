import numpy as np

from app.editing import UndoStack, fill_holes, paint_circle, replace_object, save_mask
from u_resnet_segesferoides.dataset import load_mask_labels


def _mask_with_two_blobs():
    m = np.zeros((40, 40), np.uint8)
    m[2:10, 2:10] = 2
    m[20:30, 20:30] = 2
    return m


def test_replace_object_only_touches_clicked_blob():
    m = _mask_with_two_blobs()
    assert replace_object(m, 5, 5, 3)
    assert np.all(m[2:10, 2:10] == 3)
    assert np.all(m[20:30, 20:30] == 2)


def test_replace_object_delete_and_background_noop():
    m = _mask_with_two_blobs()
    assert replace_object(m, 25, 25, 0)
    assert m[20:30, 20:30].sum() == 0
    assert not replace_object(m, 15, 15, 1)  # fondo: nada que reasignar


def test_paint_and_erase():
    m = np.zeros((20, 20), np.uint8)
    paint_circle(m, 10, 10, 3, 1)
    assert m[10, 10] == 1
    paint_circle(m, 10, 10, 3, 0)
    assert m.sum() == 0


def test_fill_holes():
    m = np.zeros((20, 20), np.uint8)
    m[2:18, 2:18] = 2
    m[8:12, 8:12] = 0
    fill_holes(m, 2)
    assert np.all(m[2:18, 2:18] == 2)
    assert m[0, 0] == 0


def test_save_mask_roundtrip(tmp_path):
    m = np.zeros((10, 10), np.uint8)
    m[0:3, 0:3] = 1
    m[4:7, 4:7] = 2
    m[8:10, 8:10] = 3
    path = str(tmp_path / "x-Mask.png")
    save_mask(path, m)
    assert np.array_equal(load_mask_labels(path), m)


def test_undo_redo():
    s = UndoStack()
    m = np.zeros((4, 4), np.uint8)
    s.push(m)
    m2 = m.copy()
    m2[0, 0] = 1
    assert np.array_equal(s.undo(m2), m)
    assert np.array_equal(s.redo(m), m2)
    assert s.undo(np.zeros((4, 4), np.uint8)) is not None
