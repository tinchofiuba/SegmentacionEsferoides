import cv2
import numpy as np

# Rutas de las imágenes
old_pred_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/pred_3classes_49 - 1d 3T3.tiff"
new_pred_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/default/49 - 1d 3T3.tiff"

def analyze_min_diameter(img_path, name):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{name}] Error: No se pudo cargar la imagen en {img_path}")
        return

    # Canal Verde es C1 (0, 255, 0)
    # BGR -> Índice 1 es Verde
    green_mask = (img[:, :, 1] > 127).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask, connectivity=8)

    min_diam = float('inf')
    max_diam = 0
    count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        diam = 2 * np.sqrt(area / np.pi)

        if diam < min_diam:
            min_diam = diam
        if diam > max_diam:
            max_diam = diam
        count += 1

    print(f"--- Análisis de {name} ---")
    print(f"Total de objetos verdes (C1) encontrados: {count}")
    if count > 0:
        print(f"Diámetro Equivalente MÍNIMO encontrado: {min_diam:.2f} px")
        print(f"Diámetro Equivalente MÁXIMO encontrado: {max_diam:.2f} px")
    else:
        print("No se encontraron objetos verdes.")
    print("-" * 30 + "\n")

analyze_min_diameter(old_pred_path, "Predicción VIEJA (pred_3classes_...)")
analyze_min_diameter(new_pred_path, "Predicción NUEVA (default/...)")
