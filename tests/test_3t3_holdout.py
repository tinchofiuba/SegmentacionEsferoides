import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train_unet import UNet, SpheroidDataset
import cv2

def overlay_mask(image, mask, color="red"):
    overlay = np.zeros_like(image)
    if color == "red":
        overlay[:, :, 0] = 1.0  
    elif color == "green":
        overlay[:, :, 1] = 1.0
        
    mask_bool = mask > 0.5
    image_copy = image.copy()
    image_copy[mask_bool] = image_copy[mask_bool] * 0.5 + overlay[mask_bool] * 0.5
    return image_copy

def test_holdout():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {DEVICE}")
    
    with open("test_3t3_holdout.json", "r") as f:
        holdout_data = json.load(f)[0]
        
    img_path = holdout_data["image"]
    mask_path = holdout_data["mask"]
    
    print(f"Evaluando imagen completa de Holdout: {os.path.basename(img_path)}")
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load("unet_3t3_finetuned_resized.pth", map_location=DEVICE, weights_only=True))
        print("Modelo unet_3t3_finetuned_resized.pth cargado exitosamente.")
    except Exception as e:
        print("Error cargando el modelo:", e)
        return
        
    model.eval()
    
    # Usar SpheroidDataset para aplicar el mismo resize que el entrenamiento
    test_dataset = SpheroidDataset([img_path], [mask_path])
    img_tensor, gt_tensor = test_dataset[0]
    
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(DEVICE))
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = pred_prob > 0.5
        
    gt_mask = gt_tensor.squeeze().cpu().numpy()
    
    # Calcular Métricas (IoU y Dice)
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    iou = intersection / union if union > 0 else 1.0
    dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum())
    
    print(f"IoU: {iou:.4f} | Dice: {dice:.4f}")
    
    # Para visualizar, usamos la imagen original pero también redimensionada a 256x256 para overlay
    # Convertimos el tensor normalizado de nuevo a un rango 0-1 RGB
    img_256 = img_tensor.squeeze().cpu().numpy()
    img_256_rgb = np.stack((img_256,)*3, axis=-1)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(img_256_rgb)
    axs[0].set_title("Imagen Redimensionada (256x256)")
    axs[0].axis('off')
    
    gt_overlay = overlay_mask(img_256_rgb, gt_mask, color="green")
    axs[1].imshow(gt_overlay)
    axs[1].set_title("Ground Truth (Verde)")
    axs[1].axis('off')
    
    pred_overlay = overlay_mask(img_256_rgb, pred_mask, color="red")
    axs[2].imshow(pred_overlay)
    axs[2].set_title(f"Predicción U-Net (IoU: {iou:.2f})")
    axs[2].axis('off')
    
    os.makedirs("Resultados_3T3", exist_ok=True)
    out_path = "Resultados_3T3/holdout_comparison_resized.jpg"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparativa guardada en {out_path}")

if __name__ == "__main__":
    test_holdout()
