"""Capa Model de la app de predicción por zona. No importa PyQt5 — recibe y
devuelve numpy arrays puros, para poder reusarse desde cualquier View
(escritorio, web, CLI) sin cambios.

Reutiliza, sin duplicar, el preprocesamiento/postprocesamiento ya validado
del paquete de entrenamiento (u_resnet_segesferoides): el "cerebro" del
modelo no se toca, esta capa solo lo orquesta sobre una imagen (o un recorte
de ella) en vez de un dataset completo.
"""

import os
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from u_resnet_segesferoides.config import load_config
from u_resnet_segesferoides.dataset import preprocess_input
from u_resnet_segesferoides.utils import (
    colorize_mask,
    find_latest_run_dir,
    pad_image,
    postprocess_mask,
    tta_inference,
)

Magnification = Literal["4x", "10x"]


@dataclass
class PredictionResult:
    pred_mask: np.ndarray       # clases 0-3, HxW, resolución nativa del crop/imagen
    color_mask: np.ndarray      # RGB, salida de colorize_mask
    overlay_rgb: np.ndarray     # imagen original + color_mask superpuesta
    class_pixel_counts: dict


def _blend_overlay(image_rgb, color_mask, alpha=0.5):
    """Superpone color_mask sobre image_rgb solo donde hay clase (fondo
    transparente), a resolución nativa — sin el resize a tamaño fijo que usa
    generate_comparison_collage (pensada para collages de 3 paneles)."""
    blended = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)
    has_class = np.any(color_mask != 0, axis=-1)
    overlay = image_rgb.copy()
    overlay[has_class] = blended[has_class]
    return overlay


class SegmentationModel:
    """Envuelve un checkpoint entrenado (un run_dir con config.yaml +
    best_model.pth) y expone predict() sobre una imagen en memoria."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.config = load_config(os.path.join(run_dir, "config.yaml"))

        model_path = os.path.join(run_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {run_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = self.config["training"]["num_classes"]

        self.model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.use_tta = self.config.get("inference", {}).get("use_tta", False)
        self.empirical_thresholds = self.config.get("inference", {}).get("empirical_thresholds")

    @classmethod
    def from_latest(cls, output_dir="outputs", model_name="insitu_3t3"):
        output_base_dir = os.path.join(output_dir, "models")
        run_dir = find_latest_run_dir(output_base_dir, model_name)
        return cls(run_dir)

    def predict(self, image_rgb, magnification: Magnification, bbox=None) -> PredictionResult:
        """`image_rgb`: array HxWx3 RGB a resolución nativa. `bbox`, si se
        pasa, es (x, y, w, h) en esa misma resolución nativa — se recorta
        ANTES de todo lo demás, así nunca se infiere sobre una versión
        reescalada (eso rompería la escala en píxeles que espera el modelo
        y los umbrales empíricos de post-proceso)."""
        if bbox is not None:
            x, y, w, h = bbox
            image_rgb = image_rgb[y : y + h, x : x + w]

        padded_img, orig_h, orig_w = pad_image(image_rgb)

        tensor_img = preprocess_input(padded_img).astype(np.float32)
        tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_tta:
                outputs = tta_inference(self.model, tensor_img)
            else:
                outputs = self.model(tensor_img)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_mask = pred_mask[:orig_h, :orig_w]

        fake_filename = f"zona_{magnification}.png"
        pred_mask = postprocess_mask(pred_mask, fake_filename, self.empirical_thresholds)

        color_mask = colorize_mask(pred_mask)
        overlay_rgb = _blend_overlay(image_rgb, color_mask)

        class_pixel_counts = {c: int(np.sum(pred_mask == c)) for c in range(4)}

        return PredictionResult(
            pred_mask=pred_mask,
            color_mask=color_mask,
            overlay_rgb=overlay_rgb,
            class_pixel_counts=class_pixel_counts,
        )
