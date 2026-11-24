"""
Convierte ImgCrudas + ImgTaggeadas (carpetas que se pasan por argumento) al formato que espera
SpheroidDataset (src/dataset.py) y los deja listos para entrenar, sin duplicar
trabajo ya hecho (re-ejecutar el script solo procesa pares nuevos o modificados).

Mapeo de color de la máscara (BGR, como lee OpenCV) a clase:
    negro   (0,   0,   0)   -> 0  fondo
    verde   (0, 255,   0)   -> 1  células sueltas
    amarillo(0, 255, 255)   -> 2  esferoides
    cyan    (255, 255,   0) -> 3  atípicos

Salida:
    <out-dir>/images/<basename>.tiff       (imagen cruda, sin cambios)
    <out-dir>/masks/<basename>-Outlined.tif (máscara de 1 canal, valores 0-3)

Uso:
    uv run python scripts/data_prep/prepare_insitu_dataset.py \
        --raw-dir <ImgCrudas> --tagged-dir <ImgTaggeadas> --out-dir <ImgTemporales>
"""
import argparse
import glob
import os

import cv2
import numpy as np

COLOR_TO_CLASS = {
    (0, 255, 0): 1,     # verde -> células sueltas
    (0, 255, 255): 2,   # amarillo -> esferoides
    (255, 255, 0): 3,   # cyan -> atípicos
}
MASK_SUFFIX = "-Mask"


def convert_mask_to_labels(tagged_path):
    img = cv2.imread(tagged_path)
    if img is None:
        return None
    labels = np.zeros(img.shape[:2], dtype=np.uint8)
    for bgr, class_id in COLOR_TO_CLASS.items():
        match = np.all(img == bgr, axis=-1)
        labels[match] = class_id
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="imágenes crudas (*.png)")
    parser.add_argument("--tagged-dir", required=True, help="máscaras a color (<base>-Mask.png)")
    parser.add_argument("--out-dir", required=True, help="carpeta de salida (images/ y masks/)")
    args = parser.parse_args()
    RAW_DIR, TAGGED_DIR, OUT_DIR = args.raw_dir, args.tagged_dir, args.out_dir
    OUT_IMAGES_DIR = os.path.join(OUT_DIR, "images")
    OUT_MASKS_DIR = os.path.join(OUT_DIR, "masks")

    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUT_MASKS_DIR, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.png")))

    procesados = 0
    saltados = 0
    faltantes = []

    for raw_path in raw_files:
        basename = os.path.splitext(os.path.basename(raw_path))[0]
        tagged_path = os.path.join(TAGGED_DIR, f"{basename}{MASK_SUFFIX}.png")

        if not os.path.exists(tagged_path):
            faltantes.append(basename)
            continue

        out_img_path = os.path.join(OUT_IMAGES_DIR, f"{basename}.tiff")
        out_mask_path = os.path.join(OUT_MASKS_DIR, f"{basename}-Outlined.tif")

        if os.path.exists(out_img_path) and os.path.exists(out_mask_path):
            saltados += 1
            continue

        if not os.path.exists(out_img_path):
            img = cv2.imread(raw_path)
            if img is None:
                print(f"[ERROR] No se pudo leer imagen cruda: {raw_path}")
                continue
            cv2.imwrite(out_img_path, img)

        if not os.path.exists(out_mask_path):
            labels = convert_mask_to_labels(tagged_path)
            if labels is None:
                print(f"[ERROR] No se pudo leer máscara: {tagged_path}")
                continue
            cv2.imwrite(out_mask_path, labels)

        procesados += 1

    print(f"Procesados (nuevos): {procesados}")
    print(f"Saltados (ya existían): {saltados}")
    if faltantes:
        print(f"Sin máscara taggeada, no se convirtieron ({len(faltantes)}):")
        for f in faltantes:
            print(f"  {f}")
    print(f"\nListo en: {OUT_DIR}")


if __name__ == "__main__":
    main()
