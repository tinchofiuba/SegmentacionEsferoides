"""Servidor de inferencia por stdin/stdout: expone un run entrenado de este
paquete a cualquier cliente que hable el "protocolo de backends de
segmentación" v1 (una petición JSON por línea → una respuesta JSON por línea).

No sabe quién es el cliente. La imagen y la máscara se intercambian por ruta
de archivo. Reusa, sin duplicar, el preprocesamiento/postprocesamiento
validado del paquete.

    uv run python -m u_resnet_segesferoides.serve --models-dir outputs/models \\
        [--model-name insitu_3t3 | --run-dir outputs/models/<run>]

Peticiones: info, list_models, load {model}, reference_stats, predict {image, magnification,
bbox, out}, shutdown. Todo lo que no sea protocolo va a stderr.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from u_resnet_segesferoides.config import load_config
from u_resnet_segesferoides.dataset import MASK_COLOR_TO_CLASS, preprocess_input
from u_resnet_segesferoides.utils import (
    find_latest_run_dir,
    pad_image,
    postprocess_mask,
    tta_inference,
)

PROTOCOL_VERSION = 1
MAGNIFICATIONS = ["4x", "10x"]
CLASS_NAMES = {1: "células sueltas", 2: "esferoides", 3: "atípicos"}


def class_definitions():
    """Clases para el cliente. Los colores salen de MASK_COLOR_TO_CLASS (que
    está en BGR, por cv2), así no se duplica el mapeo del dataset."""
    return [
        {"id": class_id, "name": CLASS_NAMES[class_id], "color": list(bgr[::-1])}
        for bgr, class_id in sorted(MASK_COLOR_TO_CLASS.items(), key=lambda kv: kv[1])
    ]


def list_runs(models_dir):
    """Runs utilizables (config.yaml + best_model.pth) bajo `models_dir`, el
    más reciente primero. `finished` (metrics.txt existe) suele ser False
    mientras se entrena: su best_model.pth se sobrescribe y es un modelo parcial."""
    if not os.path.isdir(models_dir):
        return []
    runs = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isfile(os.path.join(path, "config.yaml")) and os.path.isfile(
            os.path.join(path, "best_model.pth")
        ):
            runs.append({
                "path": os.path.abspath(path),
                "name": name,
                "finished": os.path.isfile(os.path.join(path, "metrics.txt")),
            })
    runs.sort(key=lambda r: os.path.getmtime(r["path"]), reverse=True)
    return runs


class UResnetBackend:
    """Envuelve un run (config.yaml + best_model.pth) y predice sobre una imagen."""

    def __init__(self, models_dir, initial_run_dir=None):
        self.models_dir = models_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_dir = None
        self.model = None
        if initial_run_dir:
            try:
                self.load(initial_run_dir)
            except Exception as exc:  # noqa: BLE001 - arranca sin modelo; el cliente elige
                print(f"No se pudo cargar el modelo inicial: {exc}", file=sys.stderr)

    # ---------- protocolo ----------
    def info(self):
        return {
            "protocol": PROTOCOL_VERSION,
            "name": "u-resnet",
            "classes": class_definitions(),
            "magnifications": MAGNIFICATIONS,
            "supports_bbox": True,
            "accepts_model_path": True,
            "supports_reference_stats": True,
        }

    def list_models(self):
        models = []
        for run in list_runs(self.models_dir):
            entry = {
                "id": run["path"],
                "label": run["name"],
                "current": self.run_dir is not None and run["path"] == self.run_dir,
            }
            if not run["finished"]:
                entry["note"] = "sin metrics.txt: ¿en curso?"
            models.append(entry)
        return models

    def load(self, run_dir):
        run_dir = os.path.abspath(run_dir)
        model_path = os.path.join(run_dir, "best_model.pth")
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(run_dir, "config.yaml")):
            raise FileNotFoundError(f"{run_dir} no tiene config.yaml + best_model.pth")
        config = load_config(os.path.join(run_dir, "config.yaml"))
        training = config["training"]
        model = smp.Unet(
            encoder_name=training.get("encoder_name", "resnet34"),
            encoder_weights=None,
            in_channels=3,
            classes=training["num_classes"],
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        # Recién ahora se reemplaza el estado: si algo falló arriba, queda el modelo anterior.
        self.model, self.config, self.run_dir = model, config, run_dir
        inference = config.get("inference", {})
        self.use_tta = inference.get("use_tta", False)
        self.empirical_thresholds = inference.get("empirical_thresholds")
        return {"description": os.path.basename(run_dir)}

    def reference_stats(self):
        """Estadística de entrenamiento (inference.empirical_thresholds del
        config.yaml del run) como {magnificación: {id de clase: {...}}}; las
        claves "c1", "c2"... pasan a ids. None si no hay modelo o el run no la trae."""
        if self.model is None or not self.empirical_thresholds:
            return None
        return {
            magnification: {key[1:]: stats for key, stats in per_class.items()
                            if key.startswith("c") and key[1:].isdigit()}
            for magnification, per_class in self.empirical_thresholds.items()
        }

    def predict(self, image, magnification, bbox, out):
        if self.model is None:
            raise RuntimeError("No hay ningún modelo cargado")
        if magnification not in MAGNIFICATIONS:
            raise ValueError(f"Magnificación no soportada: {magnification!r}")
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {image}")
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if bbox is not None:
            # Se recorta ANTES de inferir: nunca se reescala (rompería la escala en
            # píxeles que espera el modelo y los umbrales empíricos de post-proceso).
            x, y, w, h = (int(v) for v in bbox)
            image_rgb = image_rgb[y : y + h, x : x + w]

        pred_mask = self._infer(image_rgb, magnification)
        if not cv2.imwrite(out, pred_mask.astype(np.uint8)):
            raise OSError(f"No se pudo escribir la máscara en {out}")
        return {"mask": out}

    # ---------- inferencia (misma lógica que u-resnet-infer) ----------
    def _infer(self, image_rgb, magnification):
        padded_img, orig_h, orig_w = pad_image(image_rgb)
        tensor_img = preprocess_input(padded_img).astype(np.float32)
        tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = tta_inference(self.model, tensor_img) if self.use_tta else self.model(tensor_img)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        pred_mask = pred_mask[:orig_h, :orig_w]
        return postprocess_mask(pred_mask, f"zona_{magnification}.png", self.empirical_thresholds)


def dispatch(backend, req):
    cmd = req.get("cmd")
    if cmd == "info":
        return backend.info()
    if cmd == "list_models":
        return backend.list_models()
    if cmd == "load":
        return backend.load(req["model"])
    if cmd == "reference_stats":
        return backend.reference_stats()
    if cmd == "predict":
        return backend.predict(req["image"], req["magnification"], req.get("bbox"), req["out"])
    raise ValueError(f"comando desconocido: {cmd!r}")


def serve_loop(backend, fin, fout):
    """Atiende peticiones hasta `shutdown` o EOF. Un error en una petición se
    responde como error y no mata el proceso."""
    for line in fin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            if req.get("cmd") == "shutdown":
                fout.write(json.dumps({"id": req_id, "ok": True, "result": None}) + "\n")
                fout.flush()
                return
            resp = {"id": req_id, "ok": True, "result": dispatch(backend, req)}
        except Exception as exc:  # noqa: BLE001 - se devuelve al cliente, no se traga
            resp = {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        fout.write(json.dumps(resp) + "\n")
        fout.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models-dir", default=os.path.join("outputs", "models"))
    parser.add_argument("--model-name", help="carga al arrancar el run más reciente '<nombre>_<fecha>'")
    parser.add_argument("--run-dir", help="carga al arrancar este run concreto")
    args = parser.parse_args(argv)

    # stdout es solo del protocolo: cualquier print de una librería se desvía a stderr.
    proto_out = sys.stdout
    sys.stdout = sys.stderr

    initial = args.run_dir
    if initial is None and args.model_name:
        try:
            initial = find_latest_run_dir(args.models_dir, args.model_name)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
    backend = UResnetBackend(args.models_dir, initial)
    serve_loop(backend, sys.stdin, proto_out)


if __name__ == "__main__":
    main()
