"""Exporta el mejor checkpoint de un run (`best_model.pth`) a ONNX.

La entrada del ONNX es la imagen YA preprocesada (RGB, normalización ImageNet,
`preprocess_input`), NCHW float32, con alto y ancho dinámicos (múltiplos de 32: aplicar
`pad_image` antes). La salida son los logits (N, num_classes, H, W); el argmax, el TTA y
el post-procesado quedan por fuera del grafo.
"""

import argparse
import os

import segmentation_models_pytorch as smp
import torch
import yaml

from u_resnet_segesferoides.config import load_config

DEFAULT_OPSET = 17

# Clases con id > 0 (el 0 es fondo) y color RGB: mismo mapeo que `MASK_COLOR_TO_CLASS` (dataset.py, que está en BGR; acá en RGB).
MANIFEST_CLASSES = [
    {"id": 1, "name": "células sueltas", "color": [0, 255, 0]},
    {"id": 2, "name": "esferoides", "color": [255, 255, 0]},
    {"id": 3, "name": "atípicos", "color": [0, 255, 255]},
]


def build_manifest(config, name):
    """Manifiesto para la GUI (GUI_SegmentacionEsferoides, docs/model_manifest.md)."""
    num_classes = config["training"]["num_classes"]
    manifest = {
        "schema": 1,
        "name": name,
        "output": "semantic",
        "input": {"color": "rgb", "normalize": "imagenet", "pad_multiple": 32,
                  "tta": ["hflip", "vflip", "hvflip"]},
        "classes": [c for c in MANIFEST_CLASSES if c["id"] < num_classes],
        "magnifications": ["4x", "10x"],
        "options": [{"id": "tta", "label": "Utilizar TTA",
                     "default": bool(config.get("inference", {}).get("use_tta", True))}],
    }
    return manifest


def write_manifest(onnx_path, config, name=None):
    """Escribe `<onnx>.yaml` al lado del .onnx (mismo nombre, extensión .yaml)."""
    path = os.path.splitext(onnx_path)[0] + ".yaml"
    name = name or os.path.basename(os.path.dirname(os.path.abspath(onnx_path)))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(build_manifest(config, name), f, allow_unicode=True, sort_keys=False)
    return path


def export_run_to_onnx(run_dir, config=None, out_path=None):
    """Carga `best_model.pth` de `run_dir` y escribe `best_model.onnx` al lado."""
    config = config or load_config(os.path.join(run_dir, "config.yaml"))
    training = config["training"]
    opset = config.get("export", {}).get("onnx_opset", DEFAULT_OPSET)
    out_path = out_path or os.path.join(run_dir, "best_model.onnx")

    model = smp.Unet(
        encoder_name=training.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=training["num_classes"],
    )
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
        dynamo=False,
    )
    write_manifest(out_path, config)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exporta un run entrenado a ONNX.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(f"ONNX guardado en {export_run_to_onnx(args.run_dir)}")


if __name__ == "__main__":
    main()
