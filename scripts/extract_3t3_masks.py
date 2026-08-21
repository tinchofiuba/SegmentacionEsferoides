import os
import glob
import cv2
import numpy as np
from PIL import Image

def extract_masks():
    src_dir = "/home/martin/Descargas"
    base_out_dir = os.path.join(src_dir, "datasetsEsferoides", "3T3")
    
    img_out_dir = os.path.join(base_out_dir, "Images")
    mask_out_dir = os.path.join(base_out_dir, "Manual segmentations")
    
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    
    delimitada_files = glob.glob(os.path.join(src_dir, "*delimitada.jpg"))
    print(f"Encontradas {len(delimitada_files)} imágenes delimitadas.")
    
    for df in delimitada_files:
        base_name = os.path.basename(df)
        raw_name = base_name.replace(" delimitada.jpg", ".JPG")
        raw_path = os.path.join(src_dir, raw_name)
        
        if not os.path.exists(raw_path):
            continue
            
        print(f"Procesando {raw_name}...")
        
        img = cv2.imread(df)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        r = img_rgb[:,:,0]
        g = img_rgb[:,:,1]
        b = img_rgb[:,:,2]
        
        red_mask = (r > 150) & (g < 100) & (b < 100)
        red_mask = red_mask.astype(np.uint8) * 255
        
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask = np.zeros_like(red_mask)
        if contours:
            # FIX: Dibujar TODOS los contornos (esferoides), no solo el más grande.
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
            
        final_name = base_name.replace(" delimitada.jpg", ".tiff")
        mask_out_path = os.path.join(mask_out_dir, final_name)
        Image.fromarray(filled_mask).save(mask_out_path, format="TIFF")
        
        Image.MAX_IMAGE_PIXELS = None
        raw_img = Image.open(raw_path).convert("L")
        img_out_path = os.path.join(img_out_dir, final_name)
        raw_img.save(img_out_path, format="TIFF")
        
    print("¡Procesamiento finalizado con éxito!")

if __name__ == "__main__":
    extract_masks()
