import cv2
import numpy as np
import os

input_dir = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/0_1_2_3"

# Find the first image in the directory
image_to_process = None
for filename in os.listdir(input_dir):
    if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".png"):
        if "_testeada" not in filename: # Avoid processing an already processed one
            image_to_process = filename
            break

if image_to_process is None:
    print("No images found in the directory.")
    exit(1)

img_path = os.path.join(input_dir, image_to_process)
print(f"Reading {img_path}...")

# Read as grayscale (single channel mask with 0, 1, 2, 3)
mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("Error reading the image.")
    exit(1)

# Create a 3-channel color image for visualization
color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

# OpenCV uses BGR format
# 0 -> Black (already 0)
# 1 -> Green (B=0, G=255, R=0)
color_img[mask == 1] = (0, 255, 0)
# 2 -> Blue (B=255, G=0, R=0)
color_img[mask == 2] = (255, 0, 0)
# 3 -> White (B=255, G=255, R=255)
color_img[mask == 3] = (255, 255, 255)

# Generate output filename
basename, ext = os.path.splitext(image_to_process)
out_filename = f"{basename}_testeada{ext}"
out_path = os.path.join(input_dir, out_filename)

cv2.imwrite(out_path, color_img)
print(f"Saved visualization to {out_path}")
