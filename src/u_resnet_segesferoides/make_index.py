"""Genera `models.json`, el índice que la GUI consulta para ofrecer modelos nuevos.

    uv run u-resnet-make-index --run-dir outputs/models/<run> [--run-dir ...] --out release/

Copia `best_model.onnx` / `best_model.yaml` de cada run a `--out` como `<run>.onnx` / `<run>.yaml`
y escribe `--out/models.json` con el sha256 de cada .onnx. Después se suben los 3 archivos (o más)
a un release de GitHub (con tag fijo o `latest`); las URLs del índice son relativas, así que
funciona desde cualquier host. Formato leído por la GUI: `model_update.py` de GUI_SegmentacionEsferoides.
"""

import argparse
import hashlib
import json
import os
import shutil


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(run_dirs, out_dir, notes=None):
    os.makedirs(out_dir, exist_ok=True)
    models = []
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        onnx, manifest = (os.path.join(run_dir, f"best_model.{ext}") for ext in ("onnx", "yaml"))
        for path in (onnx, manifest):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} no existe (exportar con u-resnet-export-onnx)")
        shutil.copy2(manifest, os.path.join(out_dir, f"{name}.yaml"))
        shutil.copy2(onnx, os.path.join(out_dir, f"{name}.onnx"))
        entry = {"name": name, "onnx": f"{name}.onnx", "yaml": f"{name}.yaml", "sha256": sha256_of(onnx)}
        if notes:
            entry["notes"] = notes
        models.append(entry)
    path = os.path.join(out_dir, "models.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "models": models}, f, indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Genera el índice de modelos para la GUI.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--notes", help="texto que la GUI muestra al ofrecer el modelo")
    args = parser.parse_args()
    print(f"Índice en {build_index(args.run_dir, args.out, args.notes)}")


if __name__ == "__main__":
    main()
