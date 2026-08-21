import cv2
import numpy as np

mask_path = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/0_1_2_3/216 - 7d 3T3-Outlined.tif"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is not None:
    uniques, counts = np.unique(mask, return_counts=True)
    for u, c in zip(uniques, counts):
        print(f"Clase original {u}: {c} píxeles")
else:
    print("No se pudo cargar la máscara original.")
