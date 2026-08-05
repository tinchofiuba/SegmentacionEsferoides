import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_unet import SpheroidDataset, UNet

# --- Arquitectura del Discriminador (PatchGAN-style) ---
# El discriminador tomará la Imagen Original + Máscara (Real o Falsa)
# y decidirá si la máscara fue hecha por un humano o por la U-Net.
class Discriminator(nn.Module):
    def __init__(self, in_channels=2): # 1 canal imagen + 1 canal máscara
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False), # 256 -> 128
            *discriminator_block(64, 128),                              # 128 -> 64
            *discriminator_block(128, 256),                             # 64 -> 32
            *discriminator_block(256, 512),                             # 32 -> 16
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)       # 16 -> 15 (Salida de parches)
        )

    def forward(self, img, mask):
        # Concatenar imagen y máscara en la dimensión de los canales
        img_input = torch.cat((img, mask), 1)
        return self.model(img_input)

def train_gan():
    DATASET_ROOT = "/home/martin/Descargas/datasetsEsferoides"
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Iniciando GAN en dispositivo: {DEVICE}")
    
    # 1. Preparar datos (Reusando lógica pero usando menos imágenes para probar)
    microscopes = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    all_images, all_masks = [], []
    
    for mic in microscopes:
        img_dir = os.path.join(DATASET_ROOT, mic, "Images")
        mask_dir = os.path.join(DATASET_ROOT, mic, "Manual segmentations")
        search_pattern = os.path.join(img_dir, "**", "*.tiff")
        
        for img_path in glob.glob(search_pattern, recursive=True):
            rel_path = os.path.relpath(img_path, img_dir)
            if rel_path.endswith('.ome.tiff'):
                rel_path = rel_path.replace('.ome.tiff', '.tiff')
            mask_path = os.path.join(mask_dir, rel_path)
            
            if os.path.exists(mask_path):
                all_images.append(img_path)
                all_masks.append(mask_path)
                
    # Para la GAN, el dataset es pesado, tomaremos todo pero al estar en CPU puede demorar
    pairs = list(zip(all_images, all_masks))
    random.shuffle(pairs)
    train_size = int(0.9 * len(pairs))
    train_pairs = pairs[:train_size]
    
    train_dataset = SpheroidDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Total Entrenamiento GAN: {len(train_dataset)}")
    
    # 2. Inicializar Generador (U-Net) y Discriminador
    generator = UNet(in_channels=1, out_channels=1).to(DEVICE)
    discriminator = Discriminator(in_channels=2).to(DEVICE)
    
    # Opcional: Cargar pesos pre-entrenados del Generador para acelerar el aprendizaje
    try:
        generator.load_state_dict(torch.load("unet_multidomain_final.pth", map_location=DEVICE, weights_only=True))
        print("¡U-Net base cargada! La GAN solo se encargará de refinar los bordes.")
    except:
        print("No se encontró U-Net base, el Generador entrenará desde cero.")

    # 3. Optimizadores y Funciones de Pérdida
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    criterion_GAN = nn.BCEWithLogitsLoss() # Pérdida Adversarial (Engañar al discriminador)
    criterion_pixelwise = nn.L1Loss()      # Pérdida L1 (Mantenerse fiel al Ground Truth)
    lambda_pixel = 100                     # Peso de la pérdida L1
    
    # 4. Loop de Entrenamiento Adversarial
    for epoch in range(NUM_EPOCHS):
        generator.train()
        discriminator.train()
        
        for i, (imgs, real_masks) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            real_masks = real_masks.to(DEVICE)
            
            # --- Entrenar Generador ---
            optimizer_G.zero_grad()
            
            # El generador intenta crear una máscara falsa
            fake_masks_logits = generator(imgs)
            fake_masks_probs = torch.sigmoid(fake_masks_logits)
            
            # El discriminador evalúa la máscara falsa
            pred_fake = discriminator(imgs, fake_masks_probs)
            
            # El generador quiere que el discriminador diga que es REAL (Target = 1)
            target_real = torch.ones_like(pred_fake, device=DEVICE)
            loss_GAN = criterion_GAN(pred_fake, target_real)
            
            # El generador no debe alejarse demasiado de la forma original (L1)
            loss_pixel = criterion_pixelwise(fake_masks_probs, real_masks)
            
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            
            # --- Entrenar Discriminador ---
            optimizer_D.zero_grad()
            
            # Discriminador evalúa máscaras reales (Target = 1)
            pred_real = discriminator(imgs, real_masks)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device=DEVICE))
            
            # Discriminador evalúa máscaras falsas (Target = 0)
            # Hacemos detach() para no calcular gradientes hacia el generador acá
            pred_fake_detached = discriminator(imgs, fake_masks_probs.detach())
            loss_fake = criterion_GAN(pred_fake_detached, torch.zeros_like(pred_fake_detached, device=DEVICE))
            
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            
            if i % 20 == 0:
                print(f"[Época {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] "
                      f"[Loss D: {loss_D.item():.4f}] [Loss G: {loss_G.item():.4f}]")
                
        # Guardado
        torch.save(generator.state_dict(), f"gan_generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"gan_discriminator_epoch_{epoch+1}.pth")
        
    print("¡Entrenamiento GAN finalizado!")

if __name__ == "__main__":
    train_gan()
