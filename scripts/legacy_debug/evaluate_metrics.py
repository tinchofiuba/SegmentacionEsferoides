import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
from torchvision import transforms
from train_unet import UNet


def overlay_mask(image, mask):
    if len(image.shape) == 2:
        img_rgb = np.stack((image, image, image), axis=-1)
    else:
        img_rgb = image
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
    red_overlay = np.zeros_like(img_rgb)
    red_overlay[:, :, 0] = 1.0
    mask_bool = mask > 0.5
    img_rgb[mask_bool] = img_rgb[mask_bool] * 0.5 + red_overlay[mask_bool] * 0.5
    return img_rgb

def calculate_iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return intersection / union

def calculate_dice(pred, target):
    intersection = (pred * target).sum()
    if (pred.sum() + target.sum()) == 0:
        return 1.0
    return (2. * intersection) / (pred.sum() + target.sum())

def evaluate_metrics():
    print("Cargando rutas de test desde test_holdout_paths.json...")
    with open("test_holdout_paths.json") as f:
        test_data = json.load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    # Intentamos cargar el modelo multidominio final
    try:
        model.load_state_dict(torch.load("unet_multidomain_final.pth", map_location=DEVICE, weights_only=True))
        print("Modelo unet_multidomain_final.pth cargado exitosamente.")
    except:
        print("No se encontró el modelo final. Asegúrate de que el entrenamiento haya terminado.")
        return

    model.eval()
    resize = transforms.Resize((256, 256), antialias=True)

    total_iou = 0.0
    total_dice = 0.0

    out_dir_base = "Resultados_Test_Final"
    os.makedirs(out_dir_base, exist_ok=True)

    print(f"Evaluando {len(test_data)} imágenes nunca antes vistas y guardando visualizaciones...")

    with torch.no_grad():
        for item in test_data:
            mic = item["microscope"]
            img_path = item["image"]
            mask_path = item["mask"]

            # Cargar imagen y máscara
            image = np.array(tiff.imread(img_path), dtype=np.float32)
            mask = np.array(tiff.imread(mask_path), dtype=np.float32)

            if image.max() > 0:
                image = image / image.max()
            mask = (mask > 0).astype(np.float32)

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)
            elif len(image.shape) == 3 and image.shape[-1] <= 3:
                image = np.transpose(image, (2, 0, 1))
                if image.shape[0] == 3:
                    image = np.mean(image, axis=0, keepdims=True)

            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=0)

            image_tensor = resize(torch.tensor(image)).unsqueeze(0).to(DEVICE)
            mask_tensor = resize(torch.tensor(mask)).to(DEVICE)
            mask_tensor = (mask_tensor > 0.5).float()

            # Predicción
            pred_logits = model(image_tensor)
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > 0.5).float().squeeze()
            mask_binary = mask_tensor.squeeze()

            # Calcular métricas
            iou = calculate_iou(pred_binary.cpu().numpy(), mask_binary.cpu().numpy())
            dice = calculate_dice(pred_binary.cpu().numpy(), mask_binary.cpu().numpy())

            total_iou += iou
            total_dice += dice

            # Guardar visualización comparativa
            mic_dir = os.path.join(out_dir_base, mic)
            os.makedirs(mic_dir, exist_ok=True)

            img_plot = image_tensor.squeeze().cpu().numpy()
            pred_plot = pred_binary.cpu().numpy()
            gt_plot = mask_binary.cpu().numpy()

            combined_pred = overlay_mask(img_plot, pred_plot)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # 1. Original
            axs[0].imshow(img_plot, cmap='gray')
            axs[0].set_title("Imagen Original")
            axs[0].axis('off')

            # 2. Ground Truth
            axs[1].imshow(img_plot, cmap='gray')
            # Superponemos la máscara real usando un colormap verde y transparencia
            axs[1].imshow(np.ma.masked_where(gt_plot == 0, gt_plot), cmap='Greens', alpha=0.6, vmin=0, vmax=1)
            axs[1].set_title("Ground Truth (Esperado)")
            axs[1].axis('off')

            # 3. Predicción
            axs[2].imshow(combined_pred)
            axs[2].set_title(f"Predicción U-Net (IoU: {iou:.2f})")
            axs[2].axis('off')

            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(os.path.splitext(base_name)[0])[0] if base_name.endswith('.ome.tiff') else os.path.splitext(base_name)[0]
            out_path = os.path.join(mic_dir, f"{name_without_ext}_comparison.png")

            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    avg_iou = total_iou / len(test_data)
    avg_dice = total_dice / len(test_data)

    print("\n" + "="*40)
    print("RESULTADOS FINALES EN EL SET DE PRUEBA")
    print("="*40)
    print(f"IoU (Intersection over Union): {avg_iou:.4f}")
    print(f"Dice Coefficient:              {avg_dice:.4f}")
    print("="*40)
    if avg_iou > 0.8:
        print("¡El modelo tiene un rendimiento excelente!")
    elif avg_iou > 0.6:
        print("El modelo es decente, pero la red GAN podría ayudar a mejorar los bordes.")
    else:
        print("El modelo necesita mejoras (quizás más épocas o data augmentation).")

if __name__ == "__main__":
    evaluate_metrics()
