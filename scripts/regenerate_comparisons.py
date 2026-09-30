"""Regenera predicciones y collages de comparación para un modelo ya entrenado,
sin volver a entrenar. Usa el mismo split de test (seed fija) que train.py.
"""
import os
import sys
import argparse
import torch
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.dataset import prepare_data
from src.utils import generate_comparison_collage
from src.train import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Ej. outputs/models/insitu_3t3_2026-09-28")
    args = parser.parse_args()

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    config_path = os.path.join(args.run_dir, "config.yaml")
    checkpoint_path = os.path.join(args.run_dir, "best_model.pth")

    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    _, _, test_pairs = prepare_data(config)

    num_classes = config["training"]["num_classes"]
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for test_img, _ in test_pairs:
        run_inference(model, test_img, config, device, run_name)

    print("\n--- GENERANDO COLLAGES DE COMPARACIÓN (Original | Ground Truth | Predicción) ---")
    pred_dir = os.path.join(config["paths"]["output_dir"], "predictions", run_name)
    comparisons_dir = os.path.join(config["paths"]["output_dir"], "comparisons", run_name)
    os.makedirs(comparisons_dir, exist_ok=True)

    for test_img, gt_mask_path in test_pairs:
        basename = os.path.splitext(os.path.basename(test_img))[0]
        pred_color_path = os.path.join(pred_dir, os.path.basename(test_img))
        out_path = os.path.join(comparisons_dir, f"{basename}_collage.jpg")
        ok = generate_comparison_collage(test_img, gt_mask_path, pred_color_path, out_path)
        print(f"  [{'OK' if ok else 'ERROR'}] {basename} -> {out_path}")


if __name__ == "__main__":
    main()
