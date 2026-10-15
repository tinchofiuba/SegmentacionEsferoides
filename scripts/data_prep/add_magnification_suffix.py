"""Agrega el sufijo de aumento (_4x / _10x) al nombre de las imágenes crudas.

La magnificación de cada imagen se lee de los nombres dentro de los zips de
máscaras exportados (`<base>-Mask_4x.tif` / `<base>-Mask_10x.tif`); no se
descomprime nada. Solo renombra las imágenes crudas (`raw/`): las máscaras
(`<base>-Mask.png`) se quedan como están porque el código les quita el sufijo.

Por defecto es un dry-run; con --apply renombra y escribe el mapeo base -> aumento
en `splits/insitu_3t3/magnifications.txt` (versionado, sirve para revertir).

    uv run python scripts/data_prep/add_magnification_suffix.py \
        /media/martin/D/Mascaras-*.zip --raw-dir /media/martin/D/ImgCrudas [--apply]
"""
import argparse
import os
import re
import zipfile

NAME_RE = re.compile(r"^(?:.*/)?(?P<base>.+)-Mask_(?P<mag>4x|10x)\.tif$")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--manifest", default="splits/insitu_3t3/magnifications.txt")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    mags = {}
    for zip_path in args.zips:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                m = NAME_RE.match(name)
                if m:
                    if mags.get(m["base"], m["mag"]) != m["mag"]:
                        raise SystemExit(f"Magnificación contradictoria para {m['base']}")
                    mags[m["base"]] = m["mag"]

    renames = []
    for fname in sorted(os.listdir(args.raw_dir)):
        base, ext = os.path.splitext(fname)
        if ext != ".png" or base.endswith(("_4x", "_10x")):
            continue
        if base not in mags:
            raise SystemExit(f"Sin magnificación conocida para {fname}")
        renames.append((fname, f"{base}_{mags[base]}{ext}"))
    unused = set(mags) - {os.path.splitext(a)[0] for a, _ in renames}

    print(f"{len(renames)} imágenes a renombrar ({sum(m == '4x' for m in mags.values())} de 4x, "
          f"{sum(m == '10x' for m in mags.values())} de 10x en los zips). Sin imagen cruda: {len(unused)}")
    for old, new in renames[:5]:
        print(f"  {old} -> {new}")
    if not args.apply:
        print("Dry-run: no se renombró nada (usar --apply).")
        return

    for old, new in renames:
        os.rename(os.path.join(args.raw_dir, old), os.path.join(args.raw_dir, new))
    with open(args.manifest, "w") as f:
        for base in sorted(mags):
            f.write(f"{base}\t{mags[base]}\n")
    print(f"Renombradas. Mapeo guardado en {args.manifest}")


if __name__ == "__main__":
    main()
