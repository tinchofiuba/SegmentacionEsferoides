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

from u_resnet_segesferoides.config import load_config

DEFAULT_OPSET = 17


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
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exporta un run entrenado a ONNX.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(f"ONNX guardado en {export_run_to_onnx(args.run_dir)}")


if __name__ == "__main__":
    main()
