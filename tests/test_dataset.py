import os

import cv2
import numpy as np
import pytest

from u_resnet_segesferoides.dataset import MASK_COLOR_TO_CLASS, load_mask_labels, prepare_data


def test_load_mask_labels_decodes_known_colors(make_mask_png):
    path = make_mask_png(
        "mask.png",
        shape=(10, 10),
        class_regions=[
            ((0, 255, 0), (0, 3, 0, 3)),      # verde -> clase 1
            ((0, 255, 255), (3, 6, 0, 3)),    # amarillo -> clase 2
            ((255, 255, 0), (6, 9, 0, 3)),    # cyan -> clase 3
        ],
    )

    labels = load_mask_labels(path)

    assert labels.shape == (10, 10)
    assert (labels[0:3, 0:3] == MASK_COLOR_TO_CLASS[(0, 255, 0)]).all()
    assert (labels[3:6, 0:3] == MASK_COLOR_TO_CLASS[(0, 255, 255)]).all()
    assert (labels[6:9, 0:3] == MASK_COLOR_TO_CLASS[(255, 255, 0)]).all()
    # El resto es fondo (clase 0)
    assert (labels[:, 4:] == 0).all()


def test_load_mask_labels_unknown_color_is_background(make_mask_png):
    path = make_mask_png(
        "mask_unknown.png",
        shape=(5, 5),
        class_regions=[((123, 45, 67), (0, 5, 0, 5))],  # color que no mapea a ninguna clase
    )

    labels = load_mask_labels(path)

    assert (labels == 0).all()


def test_load_mask_labels_missing_file_returns_none(tmp_path):
    missing_path = str(tmp_path / "no_existe.png")

    assert load_mask_labels(missing_path) is None


def _write_blank_png(path, shape=(5, 5)):
    cv2.imwrite(str(path), np.zeros((*shape, 3), dtype=np.uint8))


def test_prepare_data_uses_splits_dir_when_present(tmp_path):
    images_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    splits_dir = tmp_path / "splits"
    images_dir.mkdir()
    masks_dir.mkdir()
    splits_dir.mkdir()

    basenames = ["img_a", "img_b", "img_c", "img_d"]
    for name in basenames:
        _write_blank_png(images_dir / f"{name}.png")
        _write_blank_png(masks_dir / f"{name}-Mask.png")

    (splits_dir / "train.txt").write_text("img_a\nimg_b\n")
    (splits_dir / "val.txt").write_text("img_c\n")
    (splits_dir / "test.txt").write_text("img_d\n")

    config = {
        "paths": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "splits_dir": str(splits_dir),
        },
        "training": {"seed": 42},
    }

    train_pairs, val_pairs, test_pairs = prepare_data(config)

    def names(pairs):
        return sorted(os.path.basename(p[0]).replace(".png", "") for p in pairs)

    assert names(train_pairs) == ["img_a", "img_b"]
    assert names(val_pairs) == ["img_c"]
    assert names(test_pairs) == ["img_d"]


def test_prepare_data_splits_dir_raises_on_missing_basename(tmp_path):
    images_dir = tmp_path / "raw"
    masks_dir = tmp_path / "masks"
    splits_dir = tmp_path / "splits"
    images_dir.mkdir()
    masks_dir.mkdir()
    splits_dir.mkdir()

    _write_blank_png(images_dir / "img_a.png")
    _write_blank_png(masks_dir / "img_a-Mask.png")

    (splits_dir / "train.txt").write_text("img_a\n")
    (splits_dir / "val.txt").write_text("no_existe\n")
    (splits_dir / "test.txt").write_text("")

    config = {
        "paths": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "splits_dir": str(splits_dir),
        },
        "training": {"seed": 42},
    }

    with pytest.raises(ValueError, match="no_existe"):
        prepare_data(config)
