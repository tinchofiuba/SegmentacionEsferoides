import cv2
import numpy as np

from u_resnet_segesferoides.utils import colorize_mask


def blend_mask(img, color_mask, alpha=0.6):
    mask_indices = np.any(color_mask > 0, axis=-1)
    blended = img.copy()
    blended[mask_indices] = cv2.addWeighted(img[mask_indices], 1 - alpha, color_mask[mask_indices], alpha, 0)
    return blended

def main():
    img_path = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001/216 - 7d 3T3_4x.tiff"
    gt_path = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/0_1_2_3/216 - 7d 3T3-Outlined.tif"
    pred_color_path = "/home/martin/repos/SegmentacionEsferoides/outputs/predictions/default/216 - 7d 3T3_4x.tiff"

    img = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_color = cv2.imread(pred_color_path)

    if img is None or gt_mask is None or pred_color is None:
        print("Error al cargar alguna de las imágenes.")
        return

    width = 1024
    h, w = img.shape[:2]
    ratio = width / float(w)
    new_h = int(h * ratio)

    img_res = cv2.resize(img, (width, new_h))
    pred_res = cv2.resize(pred_color, (width, new_h))
    gt_res = cv2.resize(gt_mask, (width, new_h), interpolation=cv2.INTER_NEAREST)

    gt_color = colorize_mask(gt_res)
    gt_color_bgr = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)

    gt_overlay = blend_mask(img_res, gt_color_bgr, alpha=0.6)
    pred_overlay = blend_mask(img_res, pred_res, alpha=0.6)

    collage = np.hstack((img_res, gt_overlay, pred_overlay))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(collage, "Original", (50, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(collage, "Ground Truth", (width + 50, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(collage, "Prediccion", (2 * width + 50, 50), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    out_path = "/home/martin/.gemini/antigravity/brain/4d1ee324-195e-4278-9e00-a132211c6345/artifacts/collage_comparacion.jpg"
    cv2.imwrite(out_path, collage)
    print("Collage guardado en:", out_path)

if __name__ == "__main__":
    main()
