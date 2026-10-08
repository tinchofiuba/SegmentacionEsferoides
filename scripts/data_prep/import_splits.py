"""Importa un split train/val/test de otro repo (p. ej. spheroid-seg) a este.

spheroid-seg nombra las imágenes `<nombre>_<4x|10x>` y acá se llaman
`<nombre>`: se quita el sufijo de magnificación y se verifica que no haya
colisiones ni imágenes repetidas entre splits (leak).

    uv run python scripts/data_prep/import_splits.py \\
        --src ../spheroid-seg/data/splits --dst splits/insitu_3t3
"""

import argparse
import re
from pathlib import Path

SPLITS = ("train", "val", "test")
MAG_SUFFIX = re.compile(r"_(4x|10x)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    converted = {}
    for name in SPLITS:
        lines = (Path(args.src) / f"{name}.txt").read_text().splitlines()
        stems = [MAG_SUFFIX.sub("", ln.strip()) for ln in lines if ln.strip()]
        if len(stems) != len(set(stems)):
            raise SystemExit(f"Nombres repetidos dentro de '{name}' tras quitar el sufijo")
        converted[name] = stems

    seen = {}
    for name, stems in converted.items():
        for s in stems:
            if s in seen:
                raise SystemExit(f"Leak: '{s}' está en '{seen[s]}' y en '{name}'")
            seen[s] = name

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name, stems in converted.items():
        (dst / f"{name}.txt").write_text("\n".join(stems) + "\n")
        print(f"{name}: {len(stems)} imágenes")


if __name__ == "__main__":
    main()
