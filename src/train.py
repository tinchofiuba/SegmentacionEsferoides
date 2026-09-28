import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import datetime
import shutil

# Añadir el directorio padre al sys.path para poder importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.dataset import SpheroidDataset, prepare_data, get_augmentation, preprocess_input
from src.utils import pad_image, colorize_mask, tta_inference, postprocess_mask, calculate_empirical_thresholds, generate_comparison_collage

def run_inference(model, test_img_path, config, device, model_name):
    print("\n--- EJECUTANDO INFERENCIA EN IMAGEN DE TEST ---")
    model.eval()
    basename = os.path.basename(test_img_path)
    pred_dir = os.path.join(config["paths"]["output_dir"], "predictions", model_name)
    os.makedirs(pred_dir, exist_ok=True)
    
    img = cv2.imread(test_img_path)
    if img is None:
        print("Error leyendo imagen de test.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded_img, orig_h, orig_w = pad_image(img_rgb)
    
    tensor_img = preprocess_input(padded_img).astype(np.float32)
    tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if config.get("inference", {}).get("use_tta", False):
            outputs = tta_inference(model, tensor_img)
        else:
            outputs = model(tensor_img)
            
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
    pred_mask = pred_mask[:orig_h, :orig_w]
    
    # Post-procesamiento físico empírico
    empirical_thresholds = config.get("inference", {}).get("empirical_thresholds")
    pred_mask = postprocess_mask(pred_mask, basename, empirical_thresholds)
        
    color_mask = colorize_mask(pred_mask)
    
    # Guardar sin prefijos, con el nombre real
    out_path = os.path.join(pred_dir, basename)
    # OpenCV escribe en BGR
    cv2.imwrite(out_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print(f"¡Inferencia completada! Guardado en {out_path}")

def train_model(config_path):
    config = load_config(config_path)
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    print(f"Modelo a entrenar: {model_name}")
    
    print("\n--- EXTRAYENDO UMBRALES EMPIRICOS DE LAS MÁSCARAS ---")
    empirical_th = calculate_empirical_thresholds(config)
    if "inference" not in config:
        config["inference"] = {}
    config["inference"]["empirical_thresholds"] = empirical_th
    print("Umbrales empíricos inyectados en la configuración (DNI).")
    
    train_pairs, val_pairs, test_pairs = prepare_data(config)
    
    # --- Dataset & DataLoader ---
    train_dataset = SpheroidDataset(
        [p[0] for p in train_pairs], [p[1] for p in train_pairs],
        crop_size=config["training"]["crop_size"], 
        augmentation=get_augmentation(), 
        patches_per_image=config["training"]["patches_per_image"]
    )
    val_dataset = SpheroidDataset(
        [p[0] for p in val_pairs], [p[1] for p in val_pairs],
        crop_size=config["training"]["crop_size"], 
        augmentation=None, 
        patches_per_image=config["training"]["patches_per_image"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)
    
    # --- Modelo ---
    num_classes = config["training"]["num_classes"]
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=num_classes).to(device)
    
    # --- Loss & Optimizer ---
    # Usamos Focal Loss en lugar de CrossEntropyLoss para priorizar los errores
    focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    
    def combined_loss(logits, targets): 
        return 0.5 * focal_loss(logits, targets) + 0.5 * dice_loss(logits, targets)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    
    best_val_loss = float("inf")
    best_val_iou = 0.0
    output_base_dir = os.path.join(config["paths"]["output_dir"], "models")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Crear carpeta del experimento dinamicamente
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    base_run_name = f"{model_name}_{date_str}"
    run_dir = os.path.join(output_base_dir, base_run_name)
    
    if os.path.exists(run_dir):
        counter = 1
        while os.path.exists(f"{run_dir} ({counter})"):
            counter += 1
        run_dir = f"{run_dir} ({counter})"
    os.makedirs(run_dir)
    
    # Guardar receta DNI
    import yaml
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n--- EXPERIMENTO CREADO EN: {run_dir} ---")
    
    print("\n--- INICIANDO ENTRENAMIENTO ---")
    epochs = config["training"]["epochs"]
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch") as tepoch:
            for images, masks in tepoch:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Metrics
                preds = torch.argmax(outputs, dim=1).unsqueeze(1)
                true_masks = masks.unsqueeze(1)
                tp, fp, fn, tn = smp.metrics.get_stats(preds, true_masks, mode="multiclass", num_classes=num_classes)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                train_iou += iou.item()
                
                tepoch.set_postfix(loss=loss.item(), iou=iou.item())
                
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        # VAL
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
                
                # Metrics
                preds = torch.argmax(outputs, dim=1).unsqueeze(1)
                true_masks = masks.unsqueeze(1)
                tp, fp, fn, tn = smp.metrics.get_stats(preds, true_masks, mode="multiclass", num_classes=num_classes)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                val_iou += iou.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (IoU: {avg_train_iou:.4f}) | Val Loss: {avg_val_loss:.4f} (IoU: {avg_val_iou:.4f})")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_iou = avg_val_iou
            save_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Modelo guardado en {save_path} (Val Loss={best_val_loss:.4f})")
            
    print("¡Entrenamiento finalizado exitosamente!")
    
    # Guardar métricas finales
    metrics_path = os.path.join(run_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mejor Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Val IoU en ese punto: {best_val_iou:.4f}\n")
    
    for test_img, _ in test_pairs:
        run_inference(model, test_img, config, device, model_name)

    print("\n--- GENERANDO COLLAGES DE COMPARACIÓN (Original | Ground Truth | Predicción) ---")
    pred_dir = os.path.join(config["paths"]["output_dir"], "predictions", model_name)
    comparisons_dir = os.path.join(config["paths"]["output_dir"], "comparisons", model_name)
    os.makedirs(comparisons_dir, exist_ok=True)

    for test_img, gt_mask_path in test_pairs:
        basename = os.path.splitext(os.path.basename(test_img))[0]
        pred_color_path = os.path.join(pred_dir, os.path.basename(test_img))
        out_path = os.path.join(comparisons_dir, f"{basename}_collage.jpg")
        ok = generate_comparison_collage(test_img, gt_mask_path, pred_color_path, out_path)
        print(f"  [{'OK' if ok else 'ERROR'}] {basename} -> {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Ruta al archivo de configuración YAML (ej. configs/unet_focal.yaml)")
    args = parser.parse_args()
    train_model(args.config)
