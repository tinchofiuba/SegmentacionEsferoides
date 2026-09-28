"""
Genera, para cada imagen de test, un collage de 3 paneles concatenados horizontalmente
(estilo outputs/collage_comparacion.jpg):
    1. Original          -> imagen cruda
    2. Ground Truth       -> imagen cruda + máscara real superpuesta (50% transparencia)
    3. Prediccion         -> imagen cruda + máscara predicha superpuesta (50% transparencia)

Reutiliza las predicciones YA generadas por src/train.py (colorize_mask a resolución
completa) en vez de volver a correr el modelo, así es liviano.

Uso:
    python scripts/generate_comparison_images.py --config configs/insitu_3t3.yaml
"""
import os
import glob
import argparse

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.utils import generate_comparison_collage


def main(config_path):
    config = load_config(config_path)
    model_name = os.path.splitext(os.path.basename(config_path))[0]

    images_dir = config["paths"]["images_dir"]
    masks_dir = config["paths"]["masks_dir"]
    pred_dir = os.path.join(config["paths"]["output_dir"], "predictions", model_name)
    out_dir = os.path.join(config["paths"]["output_dir"], "comparisons", model_name)
    os.makedirs(out_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.tiff")))
    if not pred_files:
        print(f"No hay predicciones en {pred_dir}. Corré primero el entrenamiento/inferencia.")
        return

    for pred_path in pred_files:
        basename = os.path.splitext(os.path.basename(pred_path))[0]
        img_path = os.path.join(images_dir, os.path.basename(pred_path))
        mask_base = basename.replace("_4x", "").replace("_10x", "")
        gt_mask_path = os.path.join(masks_dir, f"{mask_base}-Outlined.tif")

        if not os.path.exists(img_path):
            print(f"[SKIP] No se encontró la imagen original para {basename}")
            continue
        if not os.path.exists(gt_mask_path):
            print(f"[SKIP] No se encontró el ground truth para {basename}")
            continue

        out_path = os.path.join(out_dir, f"{basename}_collage.jpg")
        ok = generate_comparison_collage(img_path, gt_mask_path, pred_path, out_path)
        print(f"[{'OK' if ok else 'ERROR'}] {basename}")

    print(f"\nListo en: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/insitu_3t3.yaml")
    args = parser.parse_args()
    main(args.config)
