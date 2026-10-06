from app.geometry import compute_preview_scale, map_bbox_to_full_res


def test_compute_preview_scale_downscales_large_image():
    scale, preview_w, preview_h = compute_preview_scale(8168, 6128, max_preview_dim=1000)

    assert scale < 1.0
    assert max(preview_w, preview_h) == 1000


def test_compute_preview_scale_never_upscales_small_image():
    scale, preview_w, preview_h = compute_preview_scale(400, 300, max_preview_dim=1000)

    assert scale == 1.0
    assert (preview_w, preview_h) == (400, 300)


def test_map_bbox_to_full_res_identity_when_scale_is_one():
    bbox = map_bbox_to_full_res((10, 20, 30, 40), scale=1.0, full_w=400, full_h=300)

    assert bbox == (10, 20, 30, 40)


def test_map_bbox_to_full_res_scales_up_coordinates():
    # preview a mitad de tamaño: todo se duplica al mapear a resolución nativa
    bbox = map_bbox_to_full_res((10, 20, 30, 40), scale=0.5, full_w=1000, full_h=1000)

    assert bbox == (20, 40, 60, 80)


def test_map_bbox_to_full_res_clamps_to_image_bounds():
    bbox = map_bbox_to_full_res((90, 90, 50, 50), scale=1.0, full_w=100, full_h=100)

    x, y, w, h = bbox
    assert x + w <= 100
    assert y + h <= 100
    assert x >= 0 and y >= 0
