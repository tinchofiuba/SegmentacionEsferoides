import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from u_resnet_segesferoides.config import load_config
from u_resnet_segesferoides.utils import colorize_mask, pad_image, postprocess_mask, tta_inference


def test_single_inference():
    # Load config and model
    run_dir = "/home/martin/repos/SegmentacionEsferoides/outputs/models/default_2026-08-20 (1)"
    config_path = os.path.join(run_dir, "config.yaml")
    config = load_config(config_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = config["training"]["num_classes"]

    model_path = os.path.join(run_dir, "best_model.pth")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    img_path = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001/20 - 4d 3T3_10x.tiff"
    out_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/default/20 - 4d 3T3_10x.tiff"
    basename = os.path.basename(img_path)

    empirical_thresholds = config.get("inference", {}).get("empirical_thresholds")

    print("Corriendo modelo en una sola imagen...")
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded_img, orig_h, orig_w = pad_image(img_rgb)

    tensor_img = padded_img.astype(np.float32) / 255.0
    tensor_img = torch.from_numpy(tensor_img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = tta_inference(model, tensor_img)
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    pred_mask = pred_mask[:orig_h, :orig_w]

    # Aplicar el nuevo post-procesamiento físico
    pred_mask = postprocess_mask(pred_mask, basename, empirical_thresholds)

    color_mask = colorize_mask(pred_mask)
    cv2.imwrite(out_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print("Predicción sobrescrita con éxito aplicando la regla de agujeros.")

if __name__ == "__main__":
    test_single_inference()
