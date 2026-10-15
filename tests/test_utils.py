import os

import cv2
import numpy as np
import pytest

from u_resnet_segesferoides.utils import (
    colorize_mask,
    component_shape_metrics,
    find_latest_run_dir,
    pad_image,
    postprocess_mask,
)


def test_find_latest_run_dir_picks_most_recent_by_name(tmp_path):
    for name in ["insitu_3t3_2026-09-21", "insitu_3t3_2026-09-28", "insitu_3t3_2026-10-02"]:
        (tmp_path / name).mkdir()
    (tmp_path / "otro_modelo_2026-10-05").mkdir()

    result = find_latest_run_dir(str(tmp_path), "insitu_3t3")

    assert os.path.basename(result) == "insitu_3t3_2026-10-02"


def test_find_latest_run_dir_does_not_match_other_experiment_with_same_prefix(tmp_path):
    # "insitu_3t3_effnet_..." empieza igual que "insitu_3t3_" y alfabéticamente
    # queda después de cualquier fecha 2026-*, así que un glob ingenuo lo
    # confundiría con "el run más nuevo de insitu_3t3".
    (tmp_path / "insitu_3t3_2026-09-28").mkdir()
    (tmp_path / "insitu_3t3_2026-10-02").mkdir()
    (tmp_path / "insitu_3t3_effnet_2026-09-30").mkdir()

    result = find_latest_run_dir(str(tmp_path), "insitu_3t3")

    assert os.path.basename(result) == "insitu_3t3_2026-10-02"


def test_find_latest_run_dir_falls_back_to_run_prefix(tmp_path):
    (tmp_path / "run_0001").mkdir()
    (tmp_path / "run_0002").mkdir()

    result = find_latest_run_dir(str(tmp_path), "insitu_3t3")

    assert os.path.basename(result) == "run_0002"


def test_find_latest_run_dir_raises_when_nothing_matches(tmp_path):
    with pytest.raises(FileNotFoundError, match="insitu_3t3"):
        find_latest_run_dir(str(tmp_path), "insitu_3t3")


def test_pad_image_pads_to_multiple_of_32():
    img = np.zeros((40, 70, 3), dtype=np.uint8)

    padded, orig_h, orig_w = pad_image(img)

    assert (orig_h, orig_w) == (40, 70)
    assert padded.shape[0] % 32 == 0
    assert padded.shape[1] % 32 == 0
    assert padded.shape[0] >= 40
    assert padded.shape[1] >= 70


def test_pad_image_noop_when_already_multiple_of_32():
    img = np.zeros((64, 96, 3), dtype=np.uint8)

    padded, orig_h, orig_w = pad_image(img)

    assert padded.shape[:2] == (64, 96)
    assert (orig_h, orig_w) == (64, 96)


def test_colorize_mask_matches_known_palette():
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    color = colorize_mask(mask)

    assert tuple(color[0, 0]) == (0, 0, 0)
    assert tuple(color[0, 1]) == (0, 255, 0)
    assert tuple(color[1, 0]) == (255, 0, 0)
    assert tuple(color[1, 1]) == (255, 255, 255)


def test_postprocess_mask_without_thresholds_is_noop():
    mask = np.array([[1, 2], [3, 0]], dtype=np.uint8)

    result = postprocess_mask(mask, "image_10x.png", empirical_thresholds=None)

    assert (result == mask).all()


def _thresholds(min_diam=0.0, max_diam=999.0, min_circ=0.0):
    cls = {"n": 1, "area": {"mean": 0.0, "p5": 0.0, "p95": 0.0},
           "diameter": {"mean": 0.0, "p5": min_diam, "p95": max_diam},
           "circularity": {"mean": 0.0, "p5": min_circ, "p95": 1.0}}
    return {"10x": {"c1": cls, "c2": cls, "c3": cls}}


def test_postprocess_mask_old_format_thresholds_is_noop():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    old = {"10x": {"c1_min": 99.0, "c1_max": 100.0}}

    assert (postprocess_mask(mask, "image_10x.png", old) == mask).all()


def test_postprocess_mask_discards_blob_below_class1_p5_whatever_the_class():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:3, 0:3] = 1  # diametro ~3.4px
    mask[10:13, 10:13] = 2
    mask[20:23, 20:23] = 3

    result = postprocess_mask(mask, "image_10x.png", _thresholds(min_diam=5.0))

    assert (result == 0).all()


def test_postprocess_mask_large_circular_class1_is_kept():
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(mask, (20, 20), 12, 1, -1)

    result = postprocess_mask(mask, "image_10x.png", _thresholds(max_diam=15.0, min_circ=0.6))

    assert (result == mask).all()


def test_postprocess_mask_large_noncircular_class1_becomes_class2():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[25:30, 5:55] = 1  # barra 5x50: diametro ~17.8px, circularidad ~0.3

    result = postprocess_mask(mask, "image_10x.png", _thresholds(max_diam=10.0, min_circ=0.6))

    assert (result[25:30, 5:55] == 2).all()
    assert (result[result != 2] == 0).all()


def test_postprocess_mask_small_noncircular_class1_stays_class1():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[25:28, 5:30] = 1  # barra 3x25: poco circular pero diametro < p95

    result = postprocess_mask(mask, "image_10x.png", _thresholds(max_diam=50.0, min_circ=0.6))

    assert (result == mask).all()


def test_component_shape_metrics_circle_vs_bar():
    img = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(img, (15, 15), 10, 1, -1)
    img[40:43, 5:55] = 1
    _, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    metrics = sorted((component_shape_metrics(labels, stats, i) for i in (1, 2)), key=lambda m: m[2])

    bar, circle = metrics
    assert circle[2] > 0.8
    assert bar[2] < 0.3
