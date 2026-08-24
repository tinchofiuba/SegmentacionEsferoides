import cv2
import numpy as np
import torch

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
    color_img[mask == 1] = (0, 255, 0)      # Verde -> 1
    color_img[mask == 2] = (255, 0, 0)      # Azul -> 2
    color_img[mask == 3] = (255, 255, 255)  # Blanco -> 3
    return color_img

def tta_inference(model, tensor_img):
    """Realiza Test-Time Augmentation (4 vistas) y promedia los logits."""
    # 1. Normal
    out1 = model(tensor_img)
    
    # 2. Espejo Horizontal
    out2 = model(torch.flip(tensor_img, [3]))
    out2 = torch.flip(out2, [3])
    
    # 3. Espejo Vertical
    out3 = model(torch.flip(tensor_img, [2]))
    out3 = torch.flip(out3, [2])
    
    # 4. Espejo Horizontal y Vertical
    out4 = model(torch.flip(tensor_img, [2, 3]))
    out4 = torch.flip(out4, [2, 3])
    
    return (out1 + out2 + out3 + out4) / 4.0

def calculate_empirical_thresholds(config):
    """
    Escanea las máscaras y obtiene min/max reales (p5 y p95 del diámetro equivalente) 
    para 4x y 10x independientemente. Extrapola (x2.5) si faltan datos en alguna clase.
    """
    import glob
    import os
    
    images_dir = config["paths"]["images_dir"]
    masks_dir = config["paths"]["masks_dir"]
    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    
    diams_4x = {1: [], 2: [], 3: []}
    diams_10x = {1: [], 2: [], 3: []}
    
    for img_path in all_imgs:
        basename = os.path.basename(img_path).replace(".tiff", "")
        mask_base = basename.replace("_4x", "").replace("_10x", "")
        mask_name = mask_base + "-Outlined.tif"
        mask_path = os.path.join(masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        is_4x = "_4x" in basename
        target_dict = diams_4x if is_4x else diams_10x
        
        for c in [1, 2, 3]:
            class_mask = (mask == c).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                diam = 2 * np.sqrt(area / np.pi)
                target_dict[c].append(diam)
                
    def get_stats(diam_list):
        if len(diam_list) == 0:
            return None, None
        return float(np.percentile(diam_list, 5)), float(np.percentile(diam_list, 95))
        
    stats_4x = {c: get_stats(diams_4x[c]) for c in [1, 2, 3]}
    stats_10x = {c: get_stats(diams_10x[c]) for c in [1, 2, 3]}
    
    # Fallback extrapolation
    for c in [1, 2, 3]:
        min4, max4 = stats_4x[c]
        min10, max10 = stats_10x[c]
        
        if min4 is None and min10 is not None:
            stats_4x[c] = (min10 / 2.5, max10 / 2.5)
        elif min10 is None and min4 is not None:
            stats_10x[c] = (min4 * 2.5, max4 * 2.5)
        elif min4 is None and min10 is None:
            # Absolute fallback if class doesn't exist at all
            stats_4x[c] = (0.0, 999999.0)
            stats_10x[c] = (0.0, 999999.0)
            
    # Format into config dictionary
    thresholds_4x = {
        "c1_min": stats_4x[1][0], "c1_max": stats_4x[1][1],
        "c2_min": stats_4x[2][0], "c2_max": stats_4x[2][1],
        "c3_min": stats_4x[3][0], "c3_max": stats_4x[3][1]
    }
    
    thresholds_10x = {
        "c1_min": stats_10x[1][0], "c1_max": stats_10x[1][1],
        "c2_min": stats_10x[2][0], "c2_max": stats_10x[2][1],
        "c3_min": stats_10x[3][0], "c3_max": stats_10x[3][1]
    }
    
    # Imprimir para visualización del usuario (temporal)
    print("\n[Estadísticas Empíricas Extraídas - 4x]")
    print(f"  C1: {thresholds_4x['c1_min']:.1f} a {thresholds_4x['c1_max']:.1f}")
    print(f"  C2: {thresholds_4x['c2_min']:.1f} a {thresholds_4x['c2_max']:.1f}")
    print(f"  C3: {thresholds_4x['c3_min']:.1f} a {thresholds_4x['c3_max']:.1f}")
    
    print("\n[Estadísticas Empíricas Extraídas - 10x]")
    print(f"  C1: {thresholds_10x['c1_min']:.1f} a {thresholds_10x['c1_max']:.1f}")
    print(f"  C2: {thresholds_10x['c2_min']:.1f} a {thresholds_10x['c2_max']:.1f}")
    print(f"  C3: {thresholds_10x['c3_min']:.1f} a {thresholds_10x['c3_max']:.1f}\n")
    
    return {"4x": thresholds_4x, "10x": thresholds_10x}


def postprocess_mask(mask, filename, empirical_thresholds):
    """Limpia y corrige la máscara usando los umbrales empíricos de diámetro de su DNI."""
    if not empirical_thresholds:
        return mask # Fallback si el yaml no tiene thresholds guardados
        
    is_4x = "_4x" in filename
    th = empirical_thresholds["4x"] if is_4x else empirical_thresholds["10x"]
    
    cleaned_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    for c in [1, 2, 3]:
        class_mask = (mask == c).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            diam = 2 * np.sqrt(area / np.pi)
            
            # Filtro estricto: Descartar (dejar en 0) si es menor a su propio mínimo empírico
            if c == 1 and diam < th["c1_min"]:
                continue
            if c == 2 and diam < th["c2_min"]:
                continue
            if c == 3 and diam < th["c3_min"]:
                continue
                
            final_class = c
            
            # Regla de mutación: C1 es imposiblemente grande -> Muta a C2
            if c == 1 and diam > th["c1_max"]:
                final_class = 2
                
            # Asignar la clase final a todos los píxeles de esta mancha
            cleaned_mask[labels == i] = final_class
            
    # Rule 4: Consolidación Física (Rellenar Agujeros)
    # Rellenamos cualquier "agujero" (sea fondo o C1 atrapado) dentro de las masas mayores
    for c in [2, 3]:
        class_mask = (cleaned_mask == c).astype(np.uint8)
        if np.sum(class_mask) == 0:
            continue
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cleaned_mask, contours, -1, c, thickness=cv2.FILLED)
            
    return cleaned_mask
