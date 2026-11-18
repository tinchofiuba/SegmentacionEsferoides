"""Evalúa un modelo YOLOv8-seg con las mismas métricas que `u-resnet-eval`, sobre el
mismo split, para poder comparar modelos cabeza a cabeza.

Las instancias que devuelve YOLO se funden en una máscara semántica con los ids de
este repo (0 fondo, 1 célula suelta, 2 esferoide, 3 atípico/organoide). Donde se
solapan instancias gana la de mayor confianza. YOLO no entrega probabilidades por
píxel: no hay Brier ni log loss, ni variante post-procesada.

`ultralytics` no es dependencia del proyecto; correr con:
    uv run --with ultralytics u-resnet-eval-yolo --run-dir <dir del modelo YOLO>
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from u_resnet_segesferoides.config import load_config
from u_resnet_segesferoides.dataset import load_mask_labels, prepare_data
from u_resnet_segesferoides.evaluate import (
    CLASS_NAMES,
    Accumulator,
    group_report,
    parse_magnification,
    print_summary,
    unique_dir,
    write_csv,
)

# Índice de clase de YOLO (Celula, Esferoide, Organoide) -> id de clase de este repo.
YOLO_TO_CLASS = {0: 1, 1: 2, 2: 3}


def instances_to_semantic(masks, classes, confs, shape):
    """masks: (N, H, W) bool/0-1; devuelve máscara (H, W) uint8 con ids de este repo.
    Se pinta de menor a mayor confianza, así gana la instancia más confiable."""
    out = np.zeros(shape, dtype=np.uint8)
    for i in np.argsort(confs):
        cls = YOLO_TO_CLASS.get(int(classes[i]))
        if cls is not None:
            out[masks[i] > 0.5] = cls
    return out


def evaluate_yolo(config_path, run_dir, split="test", weights=None, imgsz=None, conf=0.25, max_images=None):
    from ultralytics import YOLO

    config = load_config(config_path)
    weights = weights or os.path.join(run_dir, "outputs", "train", "weights", "best.pt")
    info = {}
    info_path = os.path.join(run_dir, "outputs", "train", "weights", "best.pt.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    imgsz = imgsz or info.get("imgsz", 1920)

    train_pairs, val_pairs, test_pairs = prepare_data(config)
    pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}[split]
    if max_images:
        pairs = pairs[:max_images]

    model = YOLO(weights)
    class_names = [CLASS_NAMES[i] for i in range(4)]
    print(f"YOLO: {weights} | imgsz {imgsz} | conf {conf} | split {split} ({len(pairs)} imágenes) | "
          f"dataset de entrenamiento: {info.get('dataset', '?')}")

    groups = {}
    for img_path, mask_path in tqdm(pairs, desc=f"Evaluando {split} (YOLO)"):
        basename = os.path.basename(img_path)
        gt = load_mask_labels(mask_path)
        if gt is None:
            raise ValueError(f"No se pudo leer {mask_path}")
        res = model.predict(img_path, imgsz=imgsz, conf=conf, retina_masks=True, device="cpu", verbose=False)[0]
        if res.masks is None:
            pred = np.zeros(gt.shape, dtype=np.uint8)
        else:
            m = res.masks.data.cpu().numpy()
            if m.shape[1:] != gt.shape:
                raise ValueError(f"{basename}: máscara de YOLO {m.shape[1:]} != GT {gt.shape}")
            pred = instances_to_semantic(m, res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), gt.shape)
        for group in ("all", parse_magnification(basename)):
            groups.setdefault(group, Accumulator(4, with_post=False)).add(basename, gt, pred, None, None)

    report = {
        "run_dir": run_dir, "model": "yolov8-seg", "weights": weights, "split": split, "imgsz": imgsz,
        "conf": conf, "yolo_training_dataset": info.get("dataset"), "class_names": class_names,
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


def main():
    p = argparse.ArgumentParser(description="Evalúa un modelo YOLOv8-seg sobre un split.")
    p.add_argument("--config", default="configs/insitu_3t3.yaml", help="Define imágenes, máscaras y split")
    p.add_argument("--run-dir", required=True, help="Carpeta del modelo YOLO (con outputs/train/weights/best.pt)")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--weights", default=None)
    p.add_argument("--imgsz", type=int, default=None, help="Default: el de entrenamiento")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=None)
    a = p.parse_args()
    evaluate_yolo(a.config, a.run_dir, a.split, a.weights, a.imgsz, a.conf, a.max_images)


if __name__ == "__main__":
    main()
