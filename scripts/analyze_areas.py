import os
import glob
import cv2
import numpy as np

def analyze_mask_areas():
    masks_dir = "/home/martin/repos/SegmentacionEsferoides/inputsConvertidas/0_1_2_3"
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif*"))
    
    areas_c1 = []
    areas_c2 = []
    areas_c3 = []
    
    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        for c, area_list in zip([1, 2, 3], [areas_c1, areas_c2, areas_c3]):
            class_mask = (mask == c).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            
            # stats[0] is the background, so we skip it
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                area_list.append(area)
                
    print(f"Total C1 objects: {len(areas_c1)}")
    if areas_c1:
        print(f"C1 - Min: {np.min(areas_c1)}, Max: {np.max(areas_c1)}, Mean: {np.mean(areas_c1):.2f}")
        
    print(f"Total C2 objects: {len(areas_c2)}")
    if areas_c2:
        print(f"C2 - Min: {np.min(areas_c2)}, Max: {np.max(areas_c2)}, Mean: {np.mean(areas_c2):.2f}")
        
    print(f"Total C3 objects: {len(areas_c3)}")
    if areas_c3:
        print(f"C3 - Min: {np.min(areas_c3)}, Max: {np.max(areas_c3)}, Mean: {np.mean(areas_c3):.2f}")

if __name__ == "__main__":
    analyze_mask_areas()
