
import cv2
import numpy as np

orig_path = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001/8 - 4d 3T3_10x.tiff"
pred_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/default/8 - 4d 3T3_10x.png"
out_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/default/overlay_test.png"

print(f"Cargando original: {orig_path}")
orig = cv2.imread(orig_path)

print(f"Cargando predicción: {pred_path}")
pred = cv2.imread(pred_path)

if orig is None:
    print("Error: No se pudo cargar la imagen original.")
    exit(1)
if pred is None:
    print("Error: No se pudo cargar la imagen de predicción.")
    exit(1)

# 1. Redimensionar la original al tamaño estándar (1024x768)
# Usamos INTER_AREA porque achicar una imagen grande con este método evita pixelados.
orig_resized = cv2.resize(orig, (1024, 768), interpolation=cv2.INTER_AREA)

# 2. Lógica de Transparencia (Alpha Blending)
# Un alpha más bajo = más transparencia (0.15 significa 15% pintura, 85% original)
alpha = 0.15

# Creamos una copia de la original para que sea la base
overlay = orig_resized.copy()

# Mezclamos ambas imágenes
blended = cv2.addWeighted(orig_resized, 1 - alpha, pred, alpha, 0)

# 3. Mejora Visual: Solo aplicamos la transparencia donde la predicción NO es negra (fondo).
# Así el fondo se ve 100% nítido como la original, y solo las manchas brillan con color transparente.
mask = np.any(pred != [0, 0, 0], axis=-1)
overlay[mask] = blended[mask]

# 4. Guardar resultado
cv2.imwrite(out_path, overlay)
print(f"¡Éxito! Imagen fusionada guardada en: {out_path}")
