import os
import glob
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tifffile as tiff

class SpheroidDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = tiff.imread(img_path)
        mask = tiff.imread(mask_path)
        
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        
        # Normalizamos la imagen a [0, 1]
        if image.max() > 0:
            image = image / image.max()
            
        # Binarizamos la máscara
        mask = (mask > 0).astype(np.float32)
        
        # Ajustamos canales: (H, W) -> (C, H, W)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3 and image.shape[-1] <= 3:
            image = np.transpose(image, (2, 0, 1)) 
            if image.shape[0] == 3: # Si es RGB, pasar a gris
                image = np.mean(image, axis=0, keepdims=True)
            
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
            
        image = torch.tensor(image)
        mask = torch.tensor(mask)
        
        # Resize a 256x256
        resize = transforms.Resize((256, 256), antialias=True)
        image = resize(image)
        mask = resize(mask)
        mask = (mask > 0.5).float()
            
        return image, mask

# --- Arquitectura U-Net Básica ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            
            if x.shape != skip_connection.shape:
                import torch.nn.functional as F
                x = F.interpolate(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)
            
        return self.final_conv(x)

def train():
    DATASET_ROOT = "/home/martin/Descargas/datasetsEsferoides"
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_SPLIT = 0.05 # Guardaremos un 5% de cada dataset para testeo puro
    
    print(f"Dispositivo a usar: {DEVICE}")
    
    microscopes = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    
    all_train_images = []
    all_train_masks = []
    
    test_data_registry = []
    
    print("Recopilando datos y separando set de test...")
    for mic in microscopes:
        img_dir = os.path.join(DATASET_ROOT, mic, "Images")
        mask_dir = os.path.join(DATASET_ROOT, mic, "Manual segmentations")
        
        mic_images = []
        mic_masks = []
        
        search_pattern = os.path.join(img_dir, "**", "*.tiff")
        for img_path in glob.glob(search_pattern, recursive=True):
            rel_path = os.path.relpath(img_path, img_dir)
            if rel_path.endswith('.ome.tiff'):
                rel_path = rel_path.replace('.ome.tiff', '.tiff')
            mask_path = os.path.join(mask_dir, rel_path)
            
            if os.path.exists(mask_path):
                mic_images.append(img_path)
                mic_masks.append(mask_path)
                
        # Emparejar y mezclar
        pairs = list(zip(mic_images, mic_masks))
        random.shuffle(pairs)
        
        num_test = max(1, int(len(pairs) * TEST_SPLIT)) if len(pairs) > 0 else 0
        
        test_pairs = pairs[:num_test]
        train_pairs = pairs[num_test:]
        
        for img, mask in test_pairs:
            test_data_registry.append({"microscope": mic, "image": img, "mask": mask})
            
        for img, mask in train_pairs:
            all_train_images.append(img)
            all_train_masks.append(mask)
            
        print(f"{mic}: {len(train_pairs)} para entrenar/validar, {len(test_pairs)} guardadas para test.")

    # Guardar las rutas de test
    with open("test_holdout_paths.json", "w") as f:
        json.dump(test_data_registry, f, indent=4)
    print(f"-> Se guardaron {len(test_data_registry)} rutas en 'test_holdout_paths.json'.")
    
    # Split Train/Val
    pairs = list(zip(all_train_images, all_train_masks))
    random.shuffle(pairs)
    
    train_size = int(0.85 * len(pairs))
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]
    
    train_dataset = SpheroidDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs])
    val_dataset = SpheroidDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs])
    
    print(f"Total Entrenamiento: {len(train_dataset)}, Total Validación: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- LOOP DE ENTRENAMIENTO ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            predictions = model(images)
            loss = criterion(predictions, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Época {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = train_loss / len(train_loader)
        print(f"=== Época {epoch+1} Finalizada | Loss de Entrenamiento Promedio: {avg_loss:.4f} ===")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"=== Loss de Validación: {avg_val_loss:.4f} ===\n")
        
        # Guardado por época
        torch.save(model.state_dict(), f"unet_multidomain_epoch_{epoch+1}.pth")
        
    torch.save(model.state_dict(), "unet_multidomain_final.pth")
    print("Modelo final guardado en unet_multidomain_final.pth")

if __name__ == "__main__":
    train()
