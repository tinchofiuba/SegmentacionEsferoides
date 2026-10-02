
from u_resnet_segesferoides.dataset import MASK_COLOR_TO_CLASS, load_mask_labels


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
