import os
import sys
import glob
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.utils import pad_image, colorize_mask, tta_inference, postprocess_mask
from src.dataset import preprocess_input

def main(config_path):
    config = load_config(config_path)
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = config["training"]["num_classes"]
    IMAGES_DIR = config["paths"]["images_dir"]
    
    output_base_dir = os.path.join(config["paths"]["output_dir"], "models")
    
    # Encontrar la carpeta de run más reciente para ESTE modelo
    runs = sorted(glob.glob(os.path.join(output_base_dir, f"{model_name}_*")))
    if not runs:
        # Fallback a la carpeta run_* si se usó el formato viejo
        runs = sorted(glob.glob(os.path.join(output_base_dir, "run_*")))
        if not runs:
            raise FileNotFoundError(f"No se encontró ningún entrenamiento previo para el modelo '{model_name}'")
            
    run_dir = runs[-1]
            
    # Recargar la configuración desde el DNI del modelo para obtener los umbrales empíricos
    run_config_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(run_config_path):
        config = load_config(run_config_path)
            
    # Buscar el archivo del modelo
    model_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, "best_unet_3classes.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {run_dir}")
            
    # Crear carpeta de resultados limpia (predictions/nombreDelModelo/)
    pred_dir = os.path.join(config["paths"]["output_dir"], "predictions", model_name)
    os.makedirs(pred_dir, exist_ok=True)
    
    print(f"Cargando modelo desde: {model_path} en {DEVICE}")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    print(f"Encontradas {len(image_paths)} imágenes para procesar.")

    use_tta = config.get("inference", {}).get("use_tta", False)
    empirical_thresholds = config.get("inference", {}).get("empirical_thresholds")
    
    print(f"TTA Activado: {use_tta}")
    print(f"Post-procesamiento Empírico (Data-Driven): {'Activado' if empirical_thresholds else 'Desactivado'}")

    for img_path in tqdm(image_paths, desc="Procesando"):
        basename = os.path.basename(img_path)
        # Forzar extensión .png para la salida
        out_basename = os.path.splitext(basename)[0] + ".png"
        out_path = os.path.join(pred_dir, out_basename)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        padded_img, orig_h, orig_w = pad_image(img_rgb)

        tensor_img = preprocess_input(padded_img).astype(np.float32)
        tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if use_tta:
                outputs = tta_inference(model, tensor_img)
            else:
                outputs = model(tensor_img)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_mask = pred_mask[:orig_h, :orig_w]
        
        pred_mask = postprocess_mask(pred_mask, basename, empirical_thresholds)
            
        color_mask = colorize_mask(pred_mask)
        
        # Redimensionar al formato estándar (ej: 1024 x 768) antes de guardar
        color_mask_resized = cv2.resize(color_mask, (1024, 768), interpolation=cv2.INTER_NEAREST)
        orig_resized = cv2.resize(img_rgb, (1024, 768), interpolation=cv2.INTER_AREA)
        
        # Aplicar Alpha Blending selectivo
        alpha = 0.15
        overlay = orig_resized.copy()
        blended = cv2.addWeighted(orig_resized, 1 - alpha, color_mask_resized, alpha, 0)
        
        # Solo aplicamos color donde hay predicción
        mask = np.any(color_mask_resized != [0, 0, 0], axis=-1)
        overlay[mask] = blended[mask]
        
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"¡Todas las predicciones se guardaron en {pred_dir}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Ruta al archivo YAML del modelo (ej. configs/default.yaml)")
    args = parser.parse_args()
    main(args.config)
