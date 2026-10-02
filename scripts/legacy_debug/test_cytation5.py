import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from torchvision import transforms
from train_unet import UNet


def overlay_mask(image, mask):
    """
    Superpone la máscara roja sobre la imagen en escala de grises.
    """
    # Convertir imagen a RGB
    if len(image.shape) == 2:
        img_rgb = np.stack((image, image, image), axis=-1)
    else:
        img_rgb = image

    # Normalizar imagen al rango [0, 1] para matplotlib
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)

    # Crear una capa roja
    red_overlay = np.zeros_like(img_rgb)
    red_overlay[:, :, 0] = 1.0  # Canal rojo al máximo

    # Mezclar donde la máscara es 1
    mask_bool = mask > 0.5
    img_rgb[mask_bool] = img_rgb[mask_bool] * 0.5 + red_overlay[mask_bool] * 0.5

    return img_rgb

def test_cytation5():
    print("Cargando modelo...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load("unet_axiovert200.pth", map_location=DEVICE, weights_only=True))
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    model.eval()

    # Buscar imágenes
    img_dir = "/home/martin/Descargas/datasetsEsferoides/LeicaDMi3000B/Images/A549"
    search_pattern = os.path.join(img_dir, "*.tiff")
    image_paths = glob.glob(search_pattern)[:50] # Tomar solo 50

    print(f"Se evaluarán {len(image_paths)} imágenes de LeicaDMi3000B...")

    resize = transforms.Resize((256, 256), antialias=True)

    out_dir = "Resultados_Leica"
    os.makedirs(out_dir, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        # Cargar imagen
        image = tiff.imread(img_path)
        image = np.array(image, dtype=np.float32)

        # Preprocesar igual que en el entrenamiento
        if image.max() > 0:
            image = image / image.max()

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3 and image.shape[-1] <= 3:
            image = np.transpose(image, (2, 0, 1))
            if image.shape[0] == 3:
                image = np.mean(image, axis=0, keepdims=True)

        image_tensor = torch.tensor(image)
        image_tensor = resize(image_tensor)

        # Inferencia
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_logits = model(input_tensor)
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > 0.5).float()

        # Preparar para graficar
        img_plot = image_tensor.squeeze().cpu().numpy()
        pred_plot = pred_binary.squeeze().cpu().numpy()

        # Superponer
        combined = overlay_mask(img_plot, pred_plot)

        # Extraer nombre del archivo original
        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(os.path.splitext(base_name)[0])[0] if base_name.endswith('.ome.tiff') else os.path.splitext(base_name)[0]
        out_path = os.path.join(out_dir, f"{name_without_ext}_pred.png")

        # Guardar imagen individual
        plt.imsave(out_path, combined)

    print(f"¡Se guardaron las {len(image_paths)} imágenes individuales en la carpeta '{out_dir}'!")

if __name__ == "__main__":
    test_cytation5()
