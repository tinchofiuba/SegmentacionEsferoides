import glob
import os
import re
import sys

import cv2
import numpy as np
import torch


def find_latest_run_dir(output_base_dir, model_name):
    """Devuelve la carpeta de run más reciente para `model_name` (orden lexicográfico
    de nombre de carpeta: <model_name>_<fecha>, así que el último match es el más
    nuevo). Prueba primero el formato <model_name>_<fecha...>, y si no hay nada cae
    al formato viejo run_*. Levanta FileNotFoundError si no encuentra ninguno.

    El filtro exige que lo que sigue a "<model_name>_" sea un dígito (fecha), para
    no matchear accidentalmente una variante de experimento con el mismo prefijo
    (ej. "insitu_3t3_effnet_2026-09-30" al buscar "insitu_3t3").
    """
    name_pattern = re.compile(rf"^{re.escape(model_name)}_\d")
    candidates = glob.glob(os.path.join(output_base_dir, f"{model_name}_*"))
    runs = sorted(r for r in candidates if name_pattern.match(os.path.basename(r)))
    if not runs:
        runs = sorted(glob.glob(os.path.join(output_base_dir, "run_*")))
        if not runs:
            raise FileNotFoundError(
                f"No se encontró ningún entrenamiento previo para el modelo '{model_name}'"
            )
    return runs[-1]


def pad_image(img):
    """Pad image to be divisible by 32 (required by U-Net)."""
    h, w = img.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, h, w

def colorize_mask(mask):
    """Convert class indices 0,1,2,3 to RGB colors."""
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_img[mask == 1] = (0, 255, 0)      # Verde -> 1
    color_img[mask == 2] = (255, 0, 0)      # Azul -> 2
    color_img[mask == 3] = (255, 255, 255)  # Blanco -> 3
    return color_img

COLLAGE_PANEL_SIZE = (1024, 760)  # (ancho, alto) por panel
COLLAGE_ALPHA = 0.5
COLLAGE_LABELS = ["Original", "Ground Truth", "Prediccion"]


def _overlay_colored_mask(base_bgr, color_mask_rgb, alpha=COLLAGE_ALPHA):
    """Superpone color_mask_rgb (RGB, de colorize_mask) sobre base_bgr solo donde hay clase."""
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(base_bgr, 1 - alpha, color_mask_bgr, alpha, 0)
    has_class = np.any(color_mask_rgb != 0, axis=-1)
    result = base_bgr.copy()
    result[has_class] = blended[has_class]
    return result


def _add_label(img, text):
    img = img.copy()
    cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def generate_comparison_collage(img_path, gt_mask_path, pred_color_path, out_path):
    """Collage de 3 paneles (Original | Ground Truth | Prediccion) con overlay al 50%."""
    from u_resnet_segesferoides.dataset import load_mask_labels

    original = cv2.imread(img_path)
    pred_color_full = cv2.imread(pred_color_path)
    gt_labels_full = load_mask_labels(gt_mask_path)

    if original is None or pred_color_full is None or gt_labels_full is None:
        return False

    original_small = cv2.resize(original, COLLAGE_PANEL_SIZE, interpolation=cv2.INTER_AREA)
    pred_color_small_rgb = cv2.cvtColor(
        cv2.resize(pred_color_full, COLLAGE_PANEL_SIZE, interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2RGB,
    )
    gt_labels_small = cv2.resize(gt_labels_full, COLLAGE_PANEL_SIZE, interpolation=cv2.INTER_NEAREST)
    gt_color_small_rgb = colorize_mask(gt_labels_small)

    panel_original = _add_label(original_small, COLLAGE_LABELS[0])
    panel_gt = _add_label(_overlay_colored_mask(original_small, gt_color_small_rgb), COLLAGE_LABELS[1])
    panel_pred = _add_label(_overlay_colored_mask(original_small, pred_color_small_rgb), COLLAGE_LABELS[2])

    collage = cv2.hconcat([panel_original, panel_gt, panel_pred])
    cv2.imwrite(out_path, collage, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def tta_inference(model, tensor_img):
    """Realiza Test-Time Augmentation (4 vistas) y promedia los logits."""
    # 1. Normal
    out1 = model(tensor_img)

    # 2. Espejo Horizontal
    out2 = model(torch.flip(tensor_img, [3]))
    out2 = torch.flip(out2, [3])

    # 3. Espejo Vertical
    out3 = model(torch.flip(tensor_img, [2]))
    out3 = torch.flip(out3, [2])

    # 4. Espejo Horizontal y Vertical
    out4 = model(torch.flip(tensor_img, [2, 3]))
    out4 = torch.flip(out4, [2, 3])

    return (out1 + out2 + out3 + out4) / 4.0

# Factores de escala 4x -> 10x (relación lineal 10/4). Solo se usan como fallback
# cuando una clase no tiene ejemplos en una de las dos magnificaciones.
MAG_LINEAR_SCALE = 2.5
_NO_FILTER_STATS = {
    "n": 0,
    "area": {"mean": 0.0, "p5": 0.0, "p95": 999999.0},
    "diameter": {"mean": 0.0, "p5": 0.0, "p95": 999999.0},
    "circularity": {"mean": 0.0, "p5": 0.0, "p95": 1.0},
}


def component_shape_metrics(labels, stats, i):
    """Área, diámetro equivalente y circularidad (4π·A/P²) del componente `i`
    de `cv2.connectedComponentsWithStats`. El perímetro se mide sobre el contorno
    externo, recortando al bbox del componente para no tocar la imagen entera."""
    x, y, w, h, area = (int(v) for v in stats[i])
    diam = 2 * np.sqrt(area / np.pi)
    crop = (labels[max(y - 1, 0): y + h + 1, max(x - 1, 0): x + w + 1] == i).astype(np.uint8)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = max((cv2.arcLength(c, True) for c in contours), default=0.0)
    circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0
    return float(area), float(diam), float(circularity)


def _summarize(values):
    return {"mean": float(np.mean(values)), "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95))}


def _class_stats(samples):
    """samples: lista de (area, diametro, circularidad). None si está vacía."""
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    return {"n": len(samples), "area": _summarize(arr[:, 0]),
            "diameter": _summarize(arr[:, 1]), "circularity": _summarize(arr[:, 2])}


def _rescale_stats(stats, linear):
    """Pasa estadísticas de una magnificación a otra: diámetro x linear, área x linear²,
    circularidad sin cambios (adimensional)."""
    out = {"n": 0, "circularity": dict(stats["circularity"])}
    out["area"] = {k: v * linear**2 for k, v in stats["area"].items()}
    out["diameter"] = {k: v * linear for k, v in stats["diameter"].items()}
    return out


def calculate_empirical_thresholds(config):
    """
    Escanea las máscaras y obtiene, por magnificación (4x/10x) y por clase (1, 2, 3),
    el promedio, p5 y p95 de área, diámetro equivalente y circularidad.
    Si una clase no tiene ejemplos en una magnificación, se extrapola desde la otra
    (x2.5 en diámetro, x6.25 en área, circularidad igual).

    Devuelve {"4x": {"c1": {"n", "area", "diameter", "circularity"}, "c2": ..., "c3": ...}, "10x": ...}
    con cada métrica = {"mean", "p5", "p95"}.
    """
    import glob
    import os

    from u_resnet_segesferoides.dataset import load_mask_labels

    images_dir = config["paths"]["images_dir"]
    masks_dir = config["paths"]["masks_dir"]
    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    samples = {"4x": {1: [], 2: [], 3: []}, "10x": {1: [], 2: [], 3: []}}

    for img_path in all_imgs:
        basename = os.path.basename(img_path).replace(".png", "")
        mask_base = basename.replace("_4x", "").replace("_10x", "")
        mask_path = os.path.join(masks_dir, mask_base + "-Mask.png")

        if not os.path.exists(mask_path):
            continue

        mask = load_mask_labels(mask_path)
        if mask is None:
            continue

        target = samples["4x" if "_4x" in basename else "10x"]

        for c in [1, 2, 3]:
            class_mask = (mask == c).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for i in range(1, num_labels):
                target[c].append(component_shape_metrics(labels, stats, i))

    thresholds = {"4x": {}, "10x": {}}
    for c in [1, 2, 3]:
        s4, s10 = _class_stats(samples["4x"][c]), _class_stats(samples["10x"][c])
        if s4 is None and s10 is not None:
            s4 = _rescale_stats(s10, 1 / MAG_LINEAR_SCALE)
        elif s10 is None and s4 is not None:
            s10 = _rescale_stats(s4, MAG_LINEAR_SCALE)
        elif s4 is None and s10 is None:
            s4 = s10 = _NO_FILTER_STATS
        thresholds["4x"][f"c{c}"] = s4
        thresholds["10x"][f"c{c}"] = s10

    for mag in ["4x", "10x"]:
        print(f"\n[Estadísticas Empíricas Extraídas - {mag}]  (n, diámetro p5-p95, circularidad p5-p95)")
        for c in [1, 2, 3]:
            t = thresholds[mag][f"c{c}"]
            print(f"  C{c}: n={t['n']}  diam {t['diameter']['p5']:.1f}-{t['diameter']['p95']:.1f}"
                  f"  circ {t['circularity']['p5']:.3f}-{t['circularity']['p95']:.3f}")
    print()

    return thresholds


def postprocess_mask(mask, filename, empirical_thresholds):
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
                _, _, circ = component_shape_metrics(labels, stats, i)
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
