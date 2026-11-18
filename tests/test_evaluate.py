import numpy as np

from u_resnet_segesferoides.evaluate import (
    class_scores,
    confusion_matrix,
    finalize_proper_scores,
    merge_classes,
    object_scores,
    parse_magnification,
    proper_score_sums,
)


def test_parse_magnification():
    assert parse_magnification("/x/1 - 9d 3T3_4x.png") == "4x"
    assert parse_magnification("a_10x.png") == "10x"
    assert parse_magnification("a.png") == "unknown"


def test_confusion_and_scores():
    gt = np.array([[0, 1], [2, 2]])
    pred = np.array([[0, 2], [2, 3]])
    cm = confusion_matrix(gt, pred, 4)
    assert cm.sum() == 4 and cm[1, 2] == 1 and cm[2, 3] == 1
    s = class_scores(cm)
    assert s["dice"][0] == 1.0
    assert s["dice"][1] == 0.0
    assert np.isnan(class_scores(np.diag([0, 3]))["dice"][0])  # clase ausente → NaN


def test_merge_and_object():
    cm = np.array([[5, 1, 0, 0], [0, 2, 1, 0], [0, 0, 3, 1], [1, 0, 0, 4]])
    merged = merge_classes(cm, [[0], [1], [2, 3]])
    assert merged.sum() == cm.sum() and merged[2, 2] == 8
    obj = object_scores(cm)
    # TP object = todo el bloque foreground-foreground = 11; GT fg = 11 + 1 → recall 11/12
    assert obj["recall"] == 11 / 12


def test_proper_scores_perfect_and_uniform():
    gt = np.array([0, 1, 1, 2])
    perfect = np.eye(3)[gt]
    ps = finalize_proper_scores(proper_score_sums(perfect, gt))
    assert ps["brier"] == 0.0 and ps["log_loss"] < 1e-6
    uniform = np.full((4, 3), 1 / 3)
    ps = finalize_proper_scores(proper_score_sums(uniform, gt))
    assert abs(ps["log_loss"] - np.log(3)) < 1e-9
    assert abs(ps["brier"] - 2 / 3) < 1e-9
    absent = finalize_proper_scores(proper_score_sums(perfect, gt))["log_loss_per_class"]
    assert not np.isnan(absent).any()
    gt2 = np.array([0, 0])
    assert np.isnan(finalize_proper_scores(proper_score_sums(np.eye(3)[gt2], gt2))["log_loss_per_class"][1])


def test_yolo_instances_to_semantic_confidence_wins():
    from u_resnet_segesferoides.evaluate_yolo import instances_to_semantic

    a = np.zeros((4, 4))
    a[:, :3] = 1
    b = np.zeros((4, 4))
    b[:, 2:] = 1
    out = instances_to_semantic(np.stack([a, b]), np.array([0, 1]), np.array([0.9, 0.5]), (4, 4))
    assert out[0].tolist() == [1, 1, 1, 2]  # clase 0→1 (conf 0.9) pisa a clase 1→2 en la columna 2
