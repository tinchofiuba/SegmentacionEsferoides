import cv2
import numpy as np
import os

input_dir = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/ConColores"
output_dir = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/0_1_2_3"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not (filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png") or filename.endswith(".jpg")):
        continue

    img_path = os.path.join(input_dir, filename)
    out_path = os.path.join(output_dir, filename)

    print(f"Processing {filename}...")
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read {filename}.")
        continue

    # Create empty single-channel mask (defaults to 0 for black background)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # OpenCV loads images in BGR format
    # Cyan is (255, 255, 0) in BGR
    mask_cyan = (img[:,:,0] == 255) & (img[:,:,1] == 255) & (img[:,:,2] == 0)
    
    # Red is (0, 0, 255) in BGR
    mask_red = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 255)
    
    # Yellow is (0, 255, 255) in BGR
    mask_yellow = (img[:,:,0] == 0) & (img[:,:,1] == 255) & (img[:,:,2] == 255)

    # Assign multi-class labels
    mask[mask_cyan] = 1
    mask[mask_red] = 2
    mask[mask_yellow] = 3

    # Save as 1-channel image. 
    cv2.imwrite(out_path, mask)

print("All images converted to multi-class labels (0, 1, 2, 3) successfully.")
