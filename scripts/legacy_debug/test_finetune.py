import glob
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

# --- Configuración ---
IMAGES_DIR = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001"
OUTPUT_DIR = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions"
MODEL_PATH = "/home/martin/repos/SegmentacionEsferoides/outputs/best_unet_resnet34.pth"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

def pad_image(img):
    """Pad image to be divisible by 32 (required by U-Net)."""
    h, w = img.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, h, w

def colorize_mask(mask):
    """Convert class indices 0,1,2,3 to RGB colors."""
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # 1 -> Green (A)
    color_img[mask == 1] = (0, 255, 0)
    # 2 -> Blue (B)
    color_img[mask == 2] = (255, 0, 0)
    # 3 -> White (C)
    color_img[mask == 3] = (255, 255, 255)
    return color_img

def main():
    # Instanciar el mismo modelo
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, # Ya tenemos los nuestros
        in_channels=3,
        classes=4,
    )

    # Cargar pesos entrenados
    if not os.path.exists(MODEL_PATH):
        print(f"No se encontró el modelo entrenado en {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Obtener imágenes (probaremos con las 3 primeras)
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.tiff")))[:3]

    if len(image_paths) == 0:
        print("No se encontraron imágenes para testear.")
        return

    print(f"Testeando con {len(image_paths)} imágenes...")

    with torch.no_grad():
        for img_path in image_paths:
            basename = os.path.basename(img_path)
            print(f"Prediciendo {basename}...")

            # Leer imagen
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Padding
            padded_img, orig_h, orig_w = pad_image(img_rgb)

            # Preparar tensor
            tensor_img = padded_img.astype(np.float32) / 255.0
            tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            # Inferencia
            outputs = model(tensor_img)

            # Obtener clase con mayor probabilidad por píxel
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

            # Quitar padding
            pred_mask = pred_mask[:orig_h, :orig_w]

            # Colorear
            color_mask = colorize_mask(pred_mask)

            # Guardar resultados
            out_path = os.path.join(OUTPUT_DIR, f"pred_{basename}")
            cv2.imwrite(out_path, color_mask)
            print(f"Guardado en {out_path}")

    print("¡Test completado!")

if __name__ == "__main__":
    main()
