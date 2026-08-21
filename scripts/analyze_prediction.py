import cv2
import numpy as np

# Cargar la predicción coloreada
img_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/pred_3classes_216 - 7d 3T3.tiff"
img = cv2.imread(img_path)

if img is not None:
    # Contar píxeles de cada color
    # OpenCV lee en BGR
    # Verde: (0, 255, 0) -> Clase 1
    # Azul: (255, 0, 0) -> Clase 2
    # Blanco: (255, 255, 255) -> Clase 3
    # Negro: (0, 0, 0) -> Fondo
    
    green_pixels = np.sum(np.all(img == [0, 255, 0], axis=-1))
    blue_pixels = np.sum(np.all(img == [255, 0, 0], axis=-1))
    white_pixels = np.sum(np.all(img == [255, 255, 255], axis=-1))
    black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))
    
    print(f"Píxeles Verdes (Clase 1 / A): {green_pixels}")
    print(f"Píxeles Azules (Clase 2 / B): {blue_pixels}")
    print(f"Píxeles Blancos (Clase 3 / C): {white_pixels}")
    print(f"Píxeles Negros (Fondo): {black_pixels}")
else:
    print("No se pudo cargar la imagen de predicción.")
