import os

import cv2

input_dir = "/home/martin/Descargas/drive-download-20260811T151259Z-1-001/"

if not os.path.exists(input_dir):
    print(f"Directory {input_dir} does not exist.")
    exit(1)

count = 0
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        img_path = os.path.join(input_dir, filename)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {filename}")
            continue

        # Create output filename
        basename = os.path.splitext(filename)[0]
        out_filename = basename + ".tiff"
        out_path = os.path.join(input_dir, out_filename)

        # Save as tiff
        cv2.imwrite(out_path, img)
        count += 1
        print(f"Converted {filename} -> {out_filename}")

print(f"Successfully converted {count} images to .tiff")
