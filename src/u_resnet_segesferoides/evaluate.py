"""Evaluación de un run entrenado sobre un split (val/test), análoga a `spheroid_seg.eval`.

Inferencia sobre imagen completa (misma ruta que `u-resnet-infer`: padding a múltiplo
de 32, TTA según el config del run), y métricas acumuladas en streaming:

- matriz de confusión global (a nivel píxel) y Dice/IoU/precisión/recall por clase,
  agrupadas en `all` + por magnificación (4x/10x, del sufijo del nombre de archivo);
- Dice/IoU por imagen (promedio de imágenes donde la clase aparece en GT o predicción);
- métrica "object" (fondo vs. cualquier clase de primer plano) y vista de 3 clases
  (2+3 fusionadas como "agregado", como entrena spheroid-seg);
- Brier y log loss por clase, calculados sobre las probabilidades softmax (antes del
  argmax y sin post-proceso);
- las mismas matrices/métricas después del post-procesado empírico, si el run tiene
  `inference.empirical_thresholds` (las probabilidades no existen post-proceso).

Escribe en `outputs/evals/<run>_<split>/` (con reintento `(1)`, `(2)`...):
`metrics.json`, `metrics.csv`, `confusion_matrix*.csv`.
"""

import argparse
import csv
import json
import os
import re

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from u_resnet_segesferoides.config import load_config
from u_resnet_segesferoides.dataset import load_mask_labels, prepare_data, preprocess_input
from u_resnet_segesferoides.utils import (
    find_latest_run_dir,
    pad_image,
    postprocess_mask,
    tta_inference,
)

# Mismo epsilon que spheroid-seg: fijo, para que los scores sean comparables entre runs.
PROPER_SCORE_EPSILON = 1e-7
CLASS_NAMES = {0: "background", 1: "loose_cell", 2: "spheroid", 3: "atypical"}
# Filas de probabilidades por bloque al calcular Brier/log loss (acota la memoria
# en float64 para imágenes de ~6000x8000).
PROB_CHUNK_ROWS = 256


# ---------------------------------------------------------------- métricas puras
def parse_magnification(name):
    """'4x' / '10x' según el sufijo `_4x`/`_10x` del nombre; 'unknown' si no hay."""
    stem = os.path.splitext(os.path.basename(name))[0]
    match = re.search(r"_(4x|10x)$", stem)
    return match.group(1) if match else "unknown"


def confusion_matrix(gt, pred, num_classes):
    """Matriz (num_classes x num_classes), filas = GT, columnas = predicción."""
    idx = gt.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def merge_classes(cm, groups):
    """Fusiona clases de una matriz de confusión. `groups`: lista de listas de índices,
    una por clase resultante (ej. [[0], [1], [2, 3]])."""
    out = np.zeros((len(groups), len(groups)), dtype=cm.dtype)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            out[i, j] = cm[np.ix_(gi, gj)].sum()
    return out


def class_scores(cm):
    """Dice/IoU/precisión/recall por clase desde una matriz de confusión.
    NaN si el denominador es 0 (clase ausente en GT y predicción)."""
    tp = np.diag(cm).astype(np.float64)
    gt_px = cm.sum(axis=1).astype(np.float64)
    pred_px = cm.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "dice": np.where(gt_px + pred_px > 0, 2 * tp / (gt_px + pred_px), np.nan),
            "iou": np.where(gt_px + pred_px - tp > 0, tp / (gt_px + pred_px - tp), np.nan),
            "precision": np.where(pred_px > 0, tp / pred_px, np.nan),
            "recall": np.where(gt_px > 0, tp / gt_px, np.nan),
            "gt_pixels": gt_px.astype(np.int64),
            "pred_pixels": pred_px.astype(np.int64),
        }


def object_scores(cm):
    """Fondo (clase 0) vs. cualquier clase de primer plano; devuelve el score de 'object'."""
    s = class_scores(merge_classes(cm, [[0], list(range(1, cm.shape[0]))]))
    return {k: v[1] for k, v in s.items()}


def proper_score_sums(probs, gt, epsilon=PROPER_SCORE_EPSILON):
    """Sumas acumulables para Brier y log loss. `probs`: (N, C) softmax; `gt`: (N,)."""
    probs = probs.astype(np.float64)
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[gt]
    brier_pc = ((probs - one_hot) ** 2).sum(axis=0)
    nll = -np.log(np.clip(probs[np.arange(len(gt)), gt], epsilon, 1.0))
    ll_pc = np.bincount(gt, weights=nll, minlength=num_classes)
    return {
        "brier_per_class_sum": brier_pc,
        "log_loss_per_class_sum": ll_pc,
        "log_loss_per_class_count": np.bincount(gt, minlength=num_classes).astype(np.int64),
        "n_pixels": np.int64(len(gt)),
    }


def finalize_proper_scores(sums):
    """Brier (multiclase y one-vs-rest) y log loss (global y condicional a la clase GT).
    El log loss de una clase ausente en GT es NaN, no un valor inventado."""
    n = max(int(sums["n_pixels"]), 1)
    counts = sums["log_loss_per_class_count"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ll_pc = np.where(counts > 0, sums["log_loss_per_class_sum"] / np.maximum(counts, 1), np.nan)
    return {
        "brier": float(sums["brier_per_class_sum"].sum() / n),
        "brier_per_class": sums["brier_per_class_sum"] / n,
        "log_loss": float(sums["log_loss_per_class_sum"].sum() / n),
        "log_loss_per_class": ll_pc,
    }


class Accumulator:
    """Estado en streaming de un grupo de imágenes (`all` o una magnificación)."""

    def __init__(self, num_classes, with_post):
        self.k = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.cm_post = np.zeros_like(self.cm) if with_post else None
        self.sums = None
        self.per_image = []  # un dict de scores por imagen
        self.names = []

    def add(self, name, gt, pred, pred_post, prob_sums):
        cm = confusion_matrix(gt, pred, self.k)
        self.cm += cm
        if self.cm_post is not None:
            self.cm_post += confusion_matrix(gt, pred_post, self.k)
        if prob_sums is None:  # modelos sin probabilidades por píxel (ej. YOLO)
            pass
        elif self.sums is None:
            self.sums = {k: np.copy(v) for k, v in prob_sums.items()}
        else:
            for k, v in prob_sums.items():
                self.sums[k] = self.sums[k] + v
        self.per_image.append(class_scores(cm))
        self.names.append(name)


# ------------------------------------------------------------------ reporte
def _f(x):
    x = float(x)
    return None if np.isnan(x) else x


def _view_report(cm, class_names):
    s = class_scores(cm)
    rows = {}
    for i, cname in enumerate(class_names):
        rows[cname] = {k: _f(s[k][i]) if k not in ("gt_pixels", "pred_pixels") else int(s[k][i]) for k in s}
    obj = object_scores(cm)
    rows["object"] = {k: _f(v) if k not in ("gt_pixels", "pred_pixels") else int(v) for k, v in obj.items()}
    return rows


def group_report(acc, class_names):
    """Reporte de un grupo: pooled por clase, promedio por imagen, vista de 3 clases,
    post-proceso y proper scores."""
    report = {
        "n_images": len(acc.names),
        "pooled": _view_report(acc.cm, class_names),
        "confusion_matrix": acc.cm.tolist(),
    }
    # Promedio por imagen: solo imágenes donde la clase aparece en GT o predicción.
    per_img = {}
    for i, cname in enumerate(class_names):
        dice = [s["dice"][i] for s in acc.per_image]
        iou = [s["iou"][i] for s in acc.per_image]
        per_img[cname] = {
            "dice_mean": _f(np.nanmean(dice)) if not np.all(np.isnan(dice)) else None,
            "iou_mean": _f(np.nanmean(iou)) if not np.all(np.isnan(iou)) else None,
            "n_images_present": int(np.sum(~np.isnan(dice))),
        }
    report["per_image_mean"] = per_img
    if len(class_names) == 4:
        merged = merge_classes(acc.cm, [[0], [1], [2, 3]])
        report["pooled_3class"] = _view_report(merged, [class_names[0], class_names[1], "aggregate"])
    if acc.cm_post is not None:
        report["pooled_postprocessed"] = _view_report(acc.cm_post, class_names)
        report["confusion_matrix_postprocessed"] = acc.cm_post.tolist()
        if len(class_names) == 4:
            merged = merge_classes(acc.cm_post, [[0], [1], [2, 3]])
            report["pooled_3class_postprocessed"] = _view_report(
                merged, [class_names[0], class_names[1], "aggregate"]
            )
    if acc.sums is not None:
        ps = finalize_proper_scores(acc.sums)
        report["brier"] = {"all": ps["brier"], **{c: _f(ps["brier_per_class"][i]) for i, c in enumerate(class_names)}}
        report["log_loss"] = {"all": ps["log_loss"], **{c: _f(ps["log_loss_per_class"][i]) for i, c in enumerate(class_names)}}
    return report


def write_csv(path, report, class_names):
    cols = ["dice", "iou", "precision", "recall", "gt_pixels", "pred_pixels"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "variant", "class", *cols, "brier", "log_loss"])
        for group, rep in report["groups"].items():
            variants = [("raw", "pooled")]
            if "pooled_postprocessed" in rep:
                variants.append(("postprocessed", "pooled_postprocessed"))
            for variant, key in variants:
                for cname, row in rep[key].items():
                    brier = rep.get("brier", {}).get(cname) if variant == "raw" else None
                    ll = rep.get("log_loss", {}).get(cname) if variant == "raw" else None
                    w.writerow([group, variant, cname, *[row[c] for c in cols], brier, ll])
            if "brier" in rep:
                w.writerow([group, "raw", "all", *[None] * len(cols), rep["brier"]["all"], rep["log_loss"]["all"]])


def unique_dir(path):
    candidate, n = path, 0
    while os.path.exists(candidate):
        n += 1
        candidate = f"{path} ({n})"
    os.makedirs(candidate)
    return candidate


# ----------------------------------------------------------------- inferencia
def predict_probs(model, image_rgb, use_tta, device):
    """Logits promediados (TTA opcional) recortados al tamaño original → tensor (C, H, W)."""
    padded, h, w = pad_image(image_rgb)
    tensor = preprocess_input(padded).astype(np.float32)
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = tta_inference(model, tensor) if use_tta else model(tensor)
        return logits[0, :, :h, :w].cpu()


def chunked_proper_sums(logits, gt):
    """Brier/log loss sobre toda la imagen, por bloques de filas (acota memoria)."""
    total = None
    for r in range(0, gt.shape[0], PROB_CHUNK_ROWS):
        probs = torch.softmax(logits[:, r : r + PROB_CHUNK_ROWS], dim=0)
        probs = probs.permute(1, 2, 0).reshape(-1, logits.shape[0]).numpy()
        part = proper_score_sums(probs, gt[r : r + PROB_CHUNK_ROWS].ravel().astype(np.intp))
        if total is None:
            total = part
        else:
            total = {k: total[k] + part[k] for k in total}
    return total


def evaluate(config_path, split="test", run_dir=None, max_images=None):
    config = load_config(config_path)
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    models_dir = os.path.join(config["paths"]["output_dir"], "models")
    run_dir = run_dir or find_latest_run_dir(models_dir, model_name)
    # El config del run trae los umbrales empíricos y el encoder con el que se entrenó.
    run_config_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(run_config_path):
        config = load_config(run_config_path)

    model_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró best_model.pth en {run_dir}")

    train_pairs, val_pairs, test_pairs = prepare_data(config)
    pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[split]
    if max_images:
        pairs = pairs[:max_images]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = config["training"]["num_classes"]
    class_names = [CLASS_NAMES.get(i, f"class_{i}") for i in range(num_classes)]
    model = smp.Unet(
        encoder_name=config["training"].get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    inference = config.get("inference", {})
    use_tta = inference.get("use_tta", False)
    thresholds = inference.get("empirical_thresholds")
    print(f"Run: {run_dir} | split: {split} ({len(pairs)} imágenes) | TTA: {use_tta} | "
          f"post-proceso: {'sí' if thresholds else 'no'} | device: {device}")

    groups = {}

    def acc_for(name):
        if name not in groups:
            groups[name] = Accumulator(num_classes, with_post=bool(thresholds))
        return groups[name]

    for img_path, mask_path in tqdm(pairs, desc=f"Evaluando {split}"):
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        gt = load_mask_labels(mask_path)
        if img is None or gt is None:
            raise ValueError(f"No se pudo leer {img_path} o {mask_path}")
        if gt.shape != img.shape[:2]:
            raise ValueError(f"{basename}: máscara {gt.shape} y imagen {img.shape[:2]} difieren de tamaño")

        logits = predict_probs(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_tta, device)
        pred = torch.argmax(logits, dim=0).numpy().astype(np.uint8)
        prob_sums = chunked_proper_sums(logits, gt)
        pred_post = postprocess_mask(pred, basename, thresholds) if thresholds else None
        del logits

        mag = parse_magnification(basename)
        for group in ("all", mag):
            acc_for(group).add(basename, gt, pred, pred_post, prob_sums)

    report = {
        "run_dir": run_dir,
        "split": split,
        "use_tta": use_tta,
        "postprocess_applied": bool(thresholds),
        "class_names": class_names,
        "groups": {g: group_report(a, class_names) for g, a in sorted(groups.items())},
    }

    out_dir = unique_dir(os.path.join(config["paths"]["output_dir"], "evals", f"{os.path.basename(run_dir)}_{split}"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_csv(os.path.join(out_dir, "metrics.csv"), report, class_names)
    for g, a in groups.items():
        np.savetxt(os.path.join(out_dir, f"confusion_matrix_{g}.csv"), a.cm, fmt="%d", delimiter=",")

    print_summary(report)
    print(f"\nResultados en {out_dir}")
    return out_dir


def print_summary(report):
    for group, rep in report["groups"].items():
        print(f"\n== {group} ({rep['n_images']} imágenes) ==")
        print(f"{'clase':<12}{'dice':>8}{'iou':>8}{'prec':>8}{'recall':>8}{'brier':>8}")
        for cname, row in rep["pooled"].items():
            brier = rep.get("brier", {}).get(cname)
            vals = [row["dice"], row["iou"], row["precision"], row["recall"], brier]
            print(f"{cname:<12}" + "".join(f"{'-' if v is None else f'{v:.4f}':>8}" for v in vals))
        if "brier" in rep:
            print(f"Brier (multiclase) {rep['brier']['all']:.4f} | log loss {rep['log_loss']['all']:.4f}")
        if "pooled_postprocessed" in rep:
            post = rep["pooled_postprocessed"]
            print("post-proceso → " + " | ".join(f"{c}: dice {r['dice']:.4f}" for c, r in post.items() if r["dice"] is not None))


def main():
    parser = argparse.ArgumentParser(description="Evalúa un run entrenado sobre un split.")
    parser.add_argument("--config", default="configs/insitu_3t3.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--run-dir", default=None, help="Run a evaluar (default: el más reciente del config)")
    parser.add_argument("--max-images", type=int, default=None, help="Solo para pruebas rápidas")
    args = parser.parse_args()
    evaluate(args.config, args.split, args.run_dir, args.max_images)


if __name__ == "__main__":
    main()
