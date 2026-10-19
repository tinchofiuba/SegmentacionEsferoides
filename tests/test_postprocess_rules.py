"""El motor de reglas con las reglas por defecto reproduce el post-procesado fijo anterior."""
import cv2
import numpy as np
import pytest

from u_resnet_segesferoides.utils import (
    DEFAULT_POSTPROCESS,
    apply_postprocess,
    component_shape_metrics,
    postprocess_mask,
    validate_postprocess,
)

def legacy_postprocess_mask(mask, filename, empirical_thresholds):
    """Limpia la máscara con los umbrales empíricos de la clase 1 (solo 2 reglas):

    1. Descarta todo componente (de cualquier clase) con diámetro equivalente menor
       al p5 de la clase 1.
    2. Un componente de clase 1 con diámetro mayor al p95 de la clase 1 Y circularidad
       menor al p5 de la clase 1 pasa a clase 2.

    Después rellena los huecos de las clases 2 y 3.
    """
    if not empirical_thresholds:
        return mask # Fallback si el yaml no tiene thresholds guardados

    is_4x = "_4x" in filename
    th = empirical_thresholds["4x"] if is_4x else empirical_thresholds["10x"]
    if "c1" not in th:
        # Umbrales en formato viejo (c1_min/c1_max...): hay que recalcularlos con
        # scripts/postprocess/recompute_thresholds.py
        print("AVISO: umbrales empíricos en formato viejo; se omite el post-procesado.",
              file=sys.stderr)
        return mask
    min_diam = th["c1"]["diameter"]["p5"]
    max_diam = th["c1"]["diameter"]["p95"]
    min_circ = th["c1"]["circularity"]["p5"]

    cleaned_mask = np.zeros(mask.shape, dtype=np.uint8)

    for c in [1, 2, 3]:
        class_mask = (mask == c).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            diam = 2 * np.sqrt(area / np.pi)

            # Regla 1: absurdo por tamaño, sin importar la clase predicha
            if diam < min_diam:
                continue

            final_class = c

            # Regla 2: C1 grande Y poco circular -> es C2
            if c == 1 and diam > max_diam:
                _, _, circ, _ = component_shape_metrics(labels, stats, i)
                if circ < min_circ:
                    final_class = 2

            cleaned_mask[labels == i] = final_class

    # Consolidación física: rellenar agujeros dentro de las masas de clase 2 y 3
    for c in [2, 3]:
        class_mask = (cleaned_mask == c).astype(np.uint8)
        if np.sum(class_mask) == 0:
            continue
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cleaned_mask, contours, -1, c, thickness=cv2.FILLED)

    return cleaned_mask


TH = {"c1": {"area": {"p5": 30, "mean": 100, "p95": 900},
             "diameter": {"p5": 6.0, "mean": 12.0, "p95": 18.0},
             "circularity": {"p5": 0.6, "mean": 0.8, "p95": 0.95}}}


def _random_mask(seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((120, 120), np.uint8)
    for _ in range(25):
        c = int(rng.integers(1, 4))
        x, y = rng.integers(0, 100, 2)
        w, h = rng.integers(2, 40, 2)
        if rng.random() < 0.5:
            mask[y:y + h, x:x + w] = c
        else:
            cv2.ellipse(mask, (int(x) + 10, int(y) + 10), (int(w) // 2 + 1, int(h) // 2 + 1), 0, 0, 360, c, -1)
    return mask


@pytest.mark.parametrize("seed", range(8))
def test_default_rules_match_legacy_postprocess(seed):
    mask = _random_mask(seed)
    new = postprocess_mask(mask, "x_10x.png", {"10x": TH, "4x": TH})
    assert (new == legacy_postprocess_mask(mask, "x_10x.png", {"10x": TH, "4x": TH})).all()


def _cfg(*rules, fill_holes=()):
    return {"rules": list(rules), "fill_holes": list(fill_holes)}


def _rule(action="delete", cls="any", conds=(("diameter", "lt", 5),), **kw):
    return {"name": "r", "class": cls, "action": action,
            "conditions": [{"metric": m, "op": o, "value": v} for m, o, v in conds], **kw}


def test_delete_reports_discarded_with_original_class():
    mask = np.zeros((30, 30), np.uint8)
    mask[2:5, 2:5] = 2          # diámetro ~3.4: se descarta
    mask[10:25, 10:25] = 1
    cleaned, discarded = apply_postprocess(mask, _cfg(_rule()))
    assert (cleaned[2:5, 2:5] == 0).all() and (cleaned[10:25, 10:25] == 1).all()
    assert (discarded[2:5, 2:5] == 2).all() and discarded.sum() == 2 * 9


def test_first_matching_rule_wins_and_keep_protects():
    mask = np.zeros((30, 30), np.uint8)
    mask[2:5, 2:5] = 1
    cfg = _cfg(_rule("keep", cls=1), _rule())
    assert (apply_postprocess(mask, cfg)[0] == mask).all()


def test_reclass_user_value_class_filter_and_disabled():
    mask = np.zeros((40, 40), np.uint8)
    mask[2:12, 2:12] = 1        # área 100
    mask[20:30, 20:30] = 2      # área 100, otra clase
    reclass = _rule("reclass", cls=1, conds=[("area", "ge", 100)], to=3)
    cleaned, _ = apply_postprocess(mask, _cfg(reclass))
    assert (cleaned[2:12, 2:12] == 3).all() and (cleaned[20:30, 20:30] == 2).all()
    cleaned, _ = apply_postprocess(mask, _cfg({**reclass, "enabled": False}))
    assert (cleaned == mask).all()


def test_circularity_perimeter_and_reference_values():
    mask = np.zeros((60, 60), np.uint8)
    mask[25:30, 5:55] = 1       # barra: poco circular
    cfg = _cfg(_rule(conds=[("circularity", "lt", "c1.p5"), ("perimeter", "gt", 50)]))
    cleaned, discarded = apply_postprocess(mask, cfg, TH)
    assert cleaned.sum() == 0 and (discarded == 1).sum() == 250


def test_rule_with_unresolvable_reference_is_skipped():
    mask = np.zeros((20, 20), np.uint8)
    mask[2:4, 2:4] = 1
    cleaned, discarded = apply_postprocess(mask, _cfg(_rule(conds=[("diameter", "lt", "c9.p5")])), TH)
    assert (cleaned == mask).all() and not discarded.any()


def test_fill_holes_only_for_listed_classes():
    mask = np.zeros((30, 30), np.uint8)
    mask[5:25, 5:25] = 2
    mask[10:15, 10:15] = 0
    assert (apply_postprocess(mask, _cfg(fill_holes=[2]))[0][10:15, 10:15] == 2).all()
    assert (apply_postprocess(mask, _cfg(fill_holes=[3]))[0][10:15, 10:15] == 0).all()


@pytest.mark.parametrize("bad", [
    {"rules": "x"},
    {"rules": [{"conditions": [], "action": "explode"}]},
    {"rules": [{"conditions": [], "action": "reclass"}]},
    {"rules": [_rule(conds=[("weight", "lt", 1)])]},
    {"rules": [_rule(conds=[("area", "~", 1)])]},
    {"rules": [_rule(conds=[("area", "lt", "mucho")])]},
])
def test_invalid_rules_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_postprocess(bad)


def test_default_config_is_valid():
    validate_postprocess(DEFAULT_POSTPROCESS)


def test_reclass_continues_with_the_new_class():
    """Una regla posterior sobre la clase nueva se aplica a lo que otra reclasificó."""
    mask = np.zeros((60, 60), np.uint8)
    mask[28:31, 5:55] = 2                      # área 150, poco circular
    th = {"c2": {"circularity": {"p5": 0.5, "mean": 0.7, "p95": 0.9}},
          "c3": {"area": {"p5": 500, "mean": 1000, "p95": 2000}}}
    to3 = _rule("reclass", cls=2, conds=[("circularity", "lt", "c2.p5")], to=3)
    small3 = _rule(cls=3, conds=[("area", "lt", "c3.p5")])
    cleaned, discarded = apply_postprocess(mask, _cfg(to3, small3), th)
    assert cleaned.sum() == 0 and (discarded == 2).sum() == 150   # se descarta; cuenta con su clase original
    # "keep" corta la cadena: la protege de las reglas siguientes
    cleaned, _ = apply_postprocess(mask, _cfg(to3, _rule("keep", cls=3, conds=[("area", "gt", 0)]), small3), th)
    assert (cleaned[28:31, 5:55] == 3).all()
    # el orden importa: con la regla de descarte primero, aún ve clase 2 y no aplica
    cleaned, _ = apply_postprocess(mask, _cfg(small3, to3), th)
    assert (cleaned[28:31, 5:55] == 3).all()


def test_perimeter_accepts_reference_values():
    validate_postprocess({"rules": [_rule(conds=[("perimeter", "lt", "c1.p5")])]})
    mask = np.zeros((60, 60), np.uint8)
    mask[10:30, 10:30] = 1                     # perímetro ~76
    th = {"c1": {"perimeter": {"p5": 200, "mean": 300, "p95": 400}}}
    cleaned, _ = apply_postprocess(mask, _cfg(_rule(conds=[("perimeter", "lt", "c1.p5")])), th)
    assert (cleaned == 0).all()


def test_discarded_reasons_name_the_rule_and_measures():
    mask = np.zeros((60, 60), np.uint8)
    mask[10:14, 10:14] = 1                     # diámetro ~4.5
    mask[30:50, 30:50] = 1
    th = {"c1": {"diameter": {"p5": 10.0, "mean": 20.0, "p95": 30.0}}}
    rule = _rule(name="chicos", conds=[("diameter", "lt", "c1.p5")])
    reasons = []
    _, discarded = apply_postprocess(mask, _cfg(rule), th, reasons)
    assert len(reasons) == 1
    r = reasons[0]
    assert discarded[r["y"], r["x"]] == 1 and (r["x"], r["y"]) == (10, 10)
    assert r["rule"] == "chicos" and r["class"] == 1
    cond = r["conditions"][0]
    assert cond["metric"] == "diameter" and cond["op"] == "lt"
    assert cond["value"] == 10.0 and cond["reference"] == "c1.p5"
    assert abs(cond["actual"] - 2 * np.sqrt(16 / np.pi)) < 1e-6
