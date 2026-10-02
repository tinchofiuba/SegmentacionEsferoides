import glob
import os
import random

import albumentations as albu
import cv2
import numpy as np
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset

preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")

# Mapeo de color (BGR, como lee OpenCV) de las máscaras taggeadas a clase.
MASK_COLOR_TO_CLASS = {
    (0, 255, 0): 1,     # verde -> células sueltas
    (0, 255, 255): 2,   # amarillo -> esferoides
    (255, 255, 0): 3,   # cyan -> atípicos
}


def load_mask_labels(mask_path):
    """Lee una máscara taggeada a color y la decodifica a clases 0-3 (1 canal)."""
    img = cv2.imread(mask_path)
    if img is None:
        return None
    labels = np.zeros(img.shape[:2], dtype=np.uint8)
    for bgr, class_id in MASK_COLOR_TO_CLASS.items():
        match = np.all(img == bgr, axis=-1)
        labels[match] = class_id
    return labels


class SpheroidDataset(Dataset):
    """Dataset con crops centrados en objetos para evitar el class imbalance."""
    def __init__(self, image_paths, mask_paths, crop_size=512, augmentation=None, patches_per_image=16):
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.patches_per_image = patches_per_image
        self.samples = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            if not os.path.exists(mask_path): continue

            mask = load_mask_labels(mask_path)
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
        mask_full = load_mask_labels(mask_path)

        img = img_full[cy - h2:cy + h2, cx - h2:cx + h2]
        mask = mask_full[cy - h2:cy + h2, cx - h2:cx + h2]

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample["image"], sample["mask"]

        img = preprocess_input(img).astype(np.float32)
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
        # albu.ElasticTransform(alpha=1, sigma=50, p=0.3),
        # albu.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])

def prepare_data(config):
    images_dir = config["paths"]["images_dir"]
    masks_dir = config["paths"]["masks_dir"]
    seed = config["training"].get("seed", 42)

    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    pairs = []
    for img_path in all_imgs:
        basename = os.path.basename(img_path).replace(".png", "")
        # Las máscaras no tienen el sufijo de aumento
        mask_base = basename.replace("_4x", "").replace("_10x", "")
        mask_name = mask_base + "-Mask.png"
        mask_path = os.path.join(masks_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        pairs.append((img_path, mask_path))

    splits_dir = config["paths"].get("splits_dir")
    if splits_dir is not None:
        # Splits materializados (listas de basenames versionadas en archivo,
        # ver splits/<dataset>/{train,val,test}.txt) en vez de recalcularlos
        # con seed+shuffle en cada corrida. Tiene prioridad sobre el resto de
        # las estrategias de split de esta función.
        def basename_of(pair):
            return os.path.basename(pair[0]).replace(".png", "")

        pairs_by_basename = {basename_of(p): p for p in pairs}

        def load_split_file(split_name):
            split_path = os.path.join(splits_dir, f"{split_name}.txt")
            with open(split_path) as f:
                names = [line.strip() for line in f if line.strip()]
            missing = [n for n in names if n not in pairs_by_basename]
            if missing:
                raise ValueError(
                    f"No se encontraron estas imágenes del split '{split_name}' "
                    f"({split_path}): {missing}"
                )
            return [pairs_by_basename[n] for n in names]

        train_pairs = load_split_file("train")
        val_pairs = load_split_file("val")
        test_pairs = load_split_file("test")

        print(f"Total imágenes encontradas: {len(pairs)}")
        print(f"Splits cargados desde {splits_dir}")
        print(f"Imágenes de Entrenamiento: {len(train_pairs)}")
        print(f"Imágenes de Validación: {len(val_pairs)}")
        print(f"Imágenes de Test: {len(test_pairs)}")

        return train_pairs, val_pairs, test_pairs

    test_fixed_basenames = config["training"].get("test_fixed_basenames")
    test_count = config["training"].get("test_count")

    if test_fixed_basenames is not None or test_count is not None:
        # Test set chico: N imágenes fijas + relleno aleatorio hasta test_count.
        # El resto se reparte train/val con train_split/val_split (por defecto 70/30).
        test_fixed_basenames = test_fixed_basenames or []
        test_count = test_count if test_count is not None else len(test_fixed_basenames)

        def basename_of(pair):
            return os.path.basename(pair[0]).replace(".png", "")

        fixed_pairs = [p for p in pairs if basename_of(p) in test_fixed_basenames]
        found_names = {basename_of(p) for p in fixed_pairs}
        missing = [b for b in test_fixed_basenames if b not in found_names]
        if missing:
            raise ValueError(f"No se encontraron estas imágenes fijas de test: {missing}")

        remaining_pool = [p for p in pairs if basename_of(p) not in test_fixed_basenames]

        random.seed(seed)
        random.shuffle(remaining_pool)

        n_extra_test = max(0, test_count - len(fixed_pairs))
        extra_test_pairs = remaining_pool[:n_extra_test]
        rest_pairs = remaining_pool[n_extra_test:]

        test_pairs = fixed_pairs + extra_test_pairs

        rest_train_split = config["training"].get("train_split", 0.70)
        rest_val_split = config["training"].get("val_split", 0.30)
        total = rest_train_split + rest_val_split
        rest_train_split, rest_val_split = rest_train_split / total, rest_val_split / total

        split_idx = round(len(rest_pairs) * rest_train_split)
        train_pairs = rest_pairs[:split_idx]
        val_pairs = rest_pairs[split_idx:]

        print(f"Total imágenes encontradas: {len(pairs)}")
        print(f"Imágenes de Test ({len(test_pairs)}): {[basename_of(p) for p in test_pairs]}")
        print(f"Imágenes de Entrenamiento: {len(train_pairs)} ({rest_train_split:.0%} del resto)")
        print(f"Imágenes de Validación: {len(val_pairs)} ({rest_val_split:.0%} del resto)")

        return train_pairs, val_pairs, test_pairs

    train_split = config["training"].get("train_split")
    val_split = config["training"].get("val_split")
    test_split = config["training"].get("test_split")

    if train_split is not None and val_split is not None and test_split is not None:
        # Split proporcional en train/val/test sobre TODO el dataset
        random.seed(seed)
        random.shuffle(pairs)

        n = len(pairs)
        n_test = round(n * test_split)
        n_train = round(n * train_split)

        test_pairs = pairs[:n_test]
        train_pairs = pairs[n_test:n_test + n_train]
        val_pairs = pairs[n_test + n_train:]

        print(f"Total imágenes encontradas: {n}")
        print(f"Imágenes de Entrenamiento: {len(train_pairs)} ({train_split:.0%})")
        print(f"Imágenes de Validación: {len(val_pairs)} ({val_split:.0%})")
        print(f"Imágenes de Test: {len(test_pairs)} ({test_split:.0%})")

        return train_pairs, val_pairs, test_pairs

    # --- Modo legado: una sola imagen de test por nombre + 70/30 sobre el resto ---
    test_basename = config["training"]["test_image_basename"]

    test_img, test_mask = None, None
    train_val_pairs = []
    for img_path, mask_path in pairs:
        basename = os.path.basename(img_path).replace(".png", "")
        if basename == test_basename:
            test_img, test_mask = img_path, mask_path
        else:
            train_val_pairs.append((img_path, mask_path))

    random.seed(seed)
    random.shuffle(train_val_pairs)

    split_idx = int(len(train_val_pairs) * 0.70)
    train_pairs = train_val_pairs[:split_idx]
    val_pairs = train_val_pairs[split_idx:]

    print(f"Total imágenes encontradas: {len(train_val_pairs) + (1 if test_img else 0)}")
    print(f"Imagen apartada de Test: {test_img}")
    print(f"Imágenes de Entrenamiento: {len(train_pairs)}")
    print(f"Imágenes de Validación: {len(val_pairs)}")

    test_pairs = [(test_img, test_mask)] if test_img else []
    return train_pairs, val_pairs, test_pairs
