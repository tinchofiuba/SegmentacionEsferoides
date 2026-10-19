"""Recalcula los umbrales empíricos de post-procesado de un run ya entrenado.

El post-procesado no depende de los pesos, solo de `inference.empirical_thresholds`
en el `config.yaml` del run: no hace falta reentrenar. Ejecutar desde la raíz del repo
(las rutas de `paths` en el config son relativas).

    uv run python scripts/postprocess/recompute_thresholds.py outputs/models/<run> [<run2> ...] [--from-run <run con umbrales ya calculados>] [--merge]

El config.yaml original se resguarda una vez como config.pre_thresholds.yaml.
"""
import argparse
import os
import shutil

import yaml

from u_resnet_segesferoides.utils import calculate_empirical_thresholds


def merge_missing(current, new):
    """Copia a `current` las claves de `new` que no tiene (recursivo); no pisa nada."""
    for key, value in new.items():
        if key not in current:
            current[key] = value
        elif isinstance(current[key], dict) and isinstance(value, dict):
            merge_missing(current[key], value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--from-run", help="copiar los umbrales del config.yaml de este run en vez de reescanear las máscaras")
    parser.add_argument("--merge", action="store_true",
                        help="conservar las estadísticas que el run ya tiene y agregar solo las que faltan (p. ej. perimeter)")
    args = parser.parse_args()

    thresholds = None  # el escaneo es lo caro; se hace una sola vez
    if args.from_run:
        with open(os.path.join(args.from_run, "config.yaml")) as f:
            thresholds = yaml.safe_load(f)["inference"]["empirical_thresholds"]
    for run_dir in args.run_dirs:
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if thresholds is None:
            thresholds = calculate_empirical_thresholds(config)
            if not any(c["n"] for mag in thresholds.values() for c in mag.values()):
                raise SystemExit(f"No se encontraron máscaras/imágenes con {config['paths']}: "
                                 "¿está montado el disco? No se modificó nada.")
        backup = os.path.join(run_dir, "config.pre_thresholds.yaml")
        if not os.path.exists(backup):
            shutil.copy(config_path, backup)
        inference = config.setdefault("inference", {})
        if args.merge and inference.get("empirical_thresholds"):
            merge_missing(inference["empirical_thresholds"], thresholds)
        else:
            inference["empirical_thresholds"] = thresholds
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Umbrales actualizados en {config_path}")


if __name__ == "__main__":
    main()
