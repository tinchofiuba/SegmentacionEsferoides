import os
import glob
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

MODEL_PATH = "/home/martin/repos/SegmentacionEsferoides/outputs/best_unet_resnet34.pth"
MASKS_DIR = "/home/martin/Descargas/drive-download-20260812T122846Z-1-001/combined_masks"
IMAGES_DIR = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Diagnosticar las máscaras ---
print("=== DIAGNÓSTICO DE MÁSCARAS ===")
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))[:3]
for mp in mask_paths:
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    print(f"{os.path.basename(mp)} - unique values: {np.unique(m)} - shape: {m.shape} - dtype: {m.dtype}")

# --- Diagnosticar salida del modelo ---
print("\n=== DIAGNÓSTICO DEL MODELO ===")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Cargar una imagen pequeña de test
img_paths = glob.glob(os.path.join(IMAGES_DIR, "*.tiff"))[:1]
img = cv2.imread(img_paths[0])
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Usar un crop de 512x512
crop = img_rgb[:512, :512]
tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)
    pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

print(f"Predicciones únicas en el crop: {np.unique(pred)}")
print(f"% de cada clase en el crop:")
for c in range(4):
    pct = 100 * (pred == c).sum() / pred.size
    print(f"  Clase {c}: {pct:.2f}%")

print(f"\nProbabilidades promedio por clase (softmax): {probs.mean(dim=(0, 2, 3)).cpu().numpy()}")
