import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

OUTPUT_DIR = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions"
MODEL_PATH = "/home/martin/repos/SegmentacionEsferoides/outputs/best_unet_3classes.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pad_image(img):
    h, w = img.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, h, w

def colorize_mask(mask):
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_img[mask == 1] = (0, 255, 0)
    color_img[mask == 2] = (255, 0, 0)
    color_img[mask == 3] = (255, 255, 255)
    return color_img

def main():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_img_path = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001/216 - 7d 3T3.tiff"
    print(f"Leyendo: {test_img_path}")
    img = cv2.imread(test_img_path)
    if img is None:
        print("ERROR")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded_img, orig_h, orig_w = pad_image(img_rgb)
    
    tensor_img = padded_img.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
    pred_mask = pred_mask[:orig_h, :orig_w]
    color_mask = colorize_mask(pred_mask)
    
    out_path = os.path.join(OUTPUT_DIR, "pred_3classes_216 - 7d 3T3.tiff")
    cv2.imwrite(out_path, color_mask)
    print(f"¡Inferencia completada! Guardado en {out_path}")

if __name__ == "__main__":
    main()
