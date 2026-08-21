import os
import cv2
import numpy as np
import glob

masks_dir = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/0_1_2_3"
mask_files = glob.glob(os.path.join(masks_dir, "*.*"))

# Filtrar para evitar las testeadas o incorrectas
mask_files = [f for f in mask_files if "_testeada" not in f and f.endswith((".tif", ".tiff", ".png"))]

found_images = []
for f in mask_files:
    mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if mask is None: continue
    uniques = np.unique(mask)
    if 1 in uniques and 2 in uniques and 3 in uniques:
        found_images.append(os.path.basename(f))

if found_images:
    print(f"FOUND MATCHES: {found_images}")
else:
    print("NO IMAGE CONTAINS ALL 3 CLASSES.")
