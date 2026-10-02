import numpy as np

from u_resnet_segesferoides.utils import colorize_mask, pad_image, postprocess_mask


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
