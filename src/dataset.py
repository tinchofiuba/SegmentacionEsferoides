import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
import random

class SpheroidDataset(Dataset):
    """Dataset con crops centrados en objetos para evitar el class imbalance."""
    def __init__(self, image_paths, mask_paths, crop_size=512, augmentation=None, patches_per_image=16):
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.patches_per_image = patches_per_image
        self.samples = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            if not os.path.exists(mask_path): continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            
            ys, xs = np.where(mask > 0)
            
            if len(ys) == 0:
                h, w = mask.shape
                for _ in range(self.patches_per_image):
                    cy = np.random.randint(crop_size // 2, h - crop_size // 2)
                    cx = np.random.randint(crop_size // 2, w - crop_size // 2)
                    self.samples.append((img_path, mask_path, cy, cx))
            else:
                h, w = mask.shape
                obj_patches = int(self.patches_per_image * 0.8)
                rand_patches = self.patches_per_image - obj_patches
                
                indices = np.random.choice(len(ys), obj_patches)
                for idx in indices:
                    cy = np.clip(ys[idx], crop_size // 2, h - crop_size // 2)
                    cx = np.clip(xs[idx], crop_size // 2, w - crop_size // 2)
                    self.samples.append((img_path, mask_path, cy, cx))
                    
                for _ in range(rand_patches):
                    cy = np.random.randint(crop_size // 2, h - crop_size // 2)
                    cx = np.random.randint(crop_size // 2, w - crop_size // 2)
                    self.samples.append((img_path, mask_path, cy, cx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path, cy, cx = self.samples[i]
        h2 = self.crop_size // 2
        
        img_full = cv2.imread(img_path)
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img = img_full[cy - h2:cy + h2, cx - h2:cx + h2]
        mask = mask_full[cy - h2:cy + h2, cx - h2:cx + h2]
        
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample["image"], sample["mask"]
            
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).long()
        return img, mask

def get_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.GaussNoise(p=0.2),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        # Descomenta abajo para transformaciones biológicas avanzadas
        # albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        # albu.GaussianBlur(blur_limit=(3, 5), p=0.2), 
    ])

def prepare_data(config):
    images_dir = config["paths"]["images_dir"]
    masks_dir = config["paths"]["masks_dir"]
    test_basename = config["training"]["test_image_basename"]
    seed = config["training"].get("seed", 42)
    
    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.tiff")))
    
    test_img = None
    test_mask = None
    train_val_imgs = []
    train_val_masks = []
    
    for img_path in all_imgs:
        basename = os.path.basename(img_path).replace(".tiff", "")
        # Las máscaras no tienen el sufijo de aumento
        mask_base = basename.replace("_4x", "").replace("_10x", "")
        mask_name = mask_base + "-Outlined.tif"
        mask_path = os.path.join(masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        if basename == test_basename:
            test_img = img_path
            test_mask = mask_path
        else:
            train_val_imgs.append(img_path)
            train_val_masks.append(mask_path)
            
    # Mezclar y separar 70/30 (con seed para reproducibilidad)
    random.seed(seed)
    pairs = list(zip(train_val_imgs, train_val_masks))
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * 0.70)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Total imágenes encontradas: {len(pairs) + (1 if test_img else 0)}")
    print(f"Imagen apartada de Test: {test_img}")
    print(f"Imágenes de Entrenamiento: {len(train_pairs)}")
    print(f"Imágenes de Validación: {len(val_pairs)}")
    
    return train_pairs, val_pairs, test_img, test_mask
