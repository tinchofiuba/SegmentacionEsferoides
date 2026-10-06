import os

import numpy as np
import pytest

from u_resnet_segesferoides.utils import (
    colorize_mask,
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


def test_postprocess_mask_discards_blob_below_min_diameter():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:3, 0:3] = 1  # blob chico, diametro ~3.4px

    thresholds = {"10x": {
        "c1_min": 5.0, "c1_max": 100.0,
        "c2_min": 0.0, "c2_max": 999.0,
        "c3_min": 0.0, "c3_max": 999.0,
    }}

    result = postprocess_mask(mask, "image_10x.png", thresholds)

    assert (result == 0).all()


def test_postprocess_mask_mutates_oversized_class1_to_class2():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 1  # blob grande 20x20, diametro ~22.6px

    thresholds = {"10x": {
        "c1_min": 0.0, "c1_max": 15.0,
        "c2_min": 0.0, "c2_max": 999.0,
        "c3_min": 0.0, "c3_max": 999.0,
    }}

    result = postprocess_mask(mask, "image_10x.png", thresholds)

    assert (result[5:25, 5:25] == 2).all()
    assert (result[result != 2] == 0).all()
