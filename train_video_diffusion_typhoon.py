"""
Train Video Diffusion Model for Typhoon Prediction
Adapted from forecast-video-diffmodels for our 72-sample dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import json
import sys

# Add forecast-video-diffmodels to path
sys.path.append('forecast-video-diffmodels/imagen')

try:
    from imagen_pytorch import Unet3D, Imagen, ImagenTrainer
    print("[OK] Successfully imported imagen_pytorch")
except ImportError as e:
    print(f"[ERROR] Failed to import imagen_pytorch: {e}")
    print("\nPlease install requirements:")
    print("  cd forecast-video-diffmodels/imagen")
    print("  pip install -r requirements.txt")
    exit(1)

print("=" * 80)
print("VIDEO DIFFUSION MODEL - TYPHOON PREDICTION")
print("=" * 80)

# Configuration
DATA_DIR = Path("forecast-video-diffmodels/dataloader/64_FC")
OUTPUT_DIR = Path("video_diffusion_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2  # Small batch for 72 samples
NUM_EPOCHS = 100  # More epochs for small dataset
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# Custom Dataset
class TyphoonVideoDataset(Dataset):
    def __init__(self, dataloader_file):
        with open(dataloader_file, 'rb') as f:
            self.dataloader = pickle.load(f)
        print(f"  Loaded {len(self.dataloader)} sequences")
    
    def __len__(self):
        return len(self.dataloader)
    
    def __getitem__(self, idx):
        item = self.dataloader[idx]
        
        # item['frames']: future frames (12, 24, 64, 64) - target
        # item['condition']: past frames (8, 24, 64, 64) - condition
        
        # For video diffusion: concatenate past and future along time dimension
        # Total: (20, 24, 64, 64)
        past = item['condition']    # (8, 24, 64, 64)
        future = item['frames']     # (12, 24, 64, 64)
        
        # Concatenate along time dimension
        video = torch.cat([past, future], dim=0)  # (20, 24, 64, 64)
        
        return {
            'video': video,           # Full video sequence
            'condition': past,        # Past frames as conditioning
            'target': future          # Future frames to predict
        }

# Load datasets
print("\n[1] Loading datasets...")
train_dataset = TyphoonVideoDataset(DATA_DIR / 'train_dataloader.dat')
val_dataset = TyphoonVideoDataset(DATA_DIR / 'val_dataloader.dat')
test_dataset = TyphoonVideoDataset(DATA_DIR / 'test_dataloader.dat')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create Imagen Video Diffusion Model
print("\n[2] Creating Video Diffusion Model...")

# Define UNet3D (with temporal layers)
unet = Unet3D(
    dim=64,                    # Base dimension
    dim_mults=(1, 2, 4, 8),   # Channel multipliers
    channels=24,               # Input channels (ERA5 variables)
    attn_heads=8,             # Attention heads
    attn_dim_head=32,         # Attention dimension per head
    use_sparse_linear_attn=True,  # Use sparse attention for efficiency
    init_conv_to_final_conv_residual=True,
    memory_efficient=True
)

# Create Imagen (handles diffusion process)
imagen = Imagen(
    unets=unet,
    image_sizes=64,           # Spatial resolution
    timesteps=1000,           # Diffusion timesteps
    channels=24,              # ERA5 channels
)

# Create trainer
trainer = ImagenTrainer(
    imagen,
    lr=LEARNING_RATE,
    use_ema=True,             # Exponential moving average
    ema_beta=0.995,
    verbose=True
).to(DEVICE)

print(f"  Model parameters: {sum(p.numel() for p in imagen.parameters()):,}")

# Training function
def train_epoch(epoch):
    trainer.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(pbar):
        video = batch['video'].to(DEVICE)  # (B, T=20, C=24, H=64, W=64)
        
        # Rearrange to (B, C, T, H, W) for Unet3D
        video = video.permute(0, 2, 1, 3, 4)
        
        # Train on the video
        loss = trainer(
            video,
            unet_number=1,           # Using first (and only) unet
            max_batch_size=BATCH_SIZE
        )
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Update model
        trainer.update(unet_number=1)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Validation function
def validate():
    trainer.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            video = batch['video'].to(DEVICE)
            video = video.permute(0, 2, 1, 3, 4)
            
            loss = trainer(
                video,
                unet_number=1,
                max_batch_size=BATCH_SIZE
            )
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Training loop
print("\n[3] Training Video Diffusion Model...")
print("=" * 80)

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # Train
    train_loss = train_epoch(epoch)
    
    # Validate
    val_loss = validate()
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trainer.save(OUTPUT_DIR / 'best_video_diffusion.pt')
        print(f"  [BEST] Saved model (val_loss: {val_loss:.4f})")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        trainer.save(OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pt')
        print(f"  [CHECKPOINT] Saved at epoch {epoch+1}")

# Save history
with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {OUTPUT_DIR / 'best_video_diffusion.pt'}")
print(f"History saved to: {OUTPUT_DIR / 'training_history.json'}")

# Generate sample predictions
print("\n[4] Generating sample predictions...")
trainer.load(OUTPUT_DIR / 'best_video_diffusion.pt')

test_sample = test_dataset[0]
condition = test_sample['condition'].unsqueeze(0).to(DEVICE)  # (1, 8, 24, 64, 64)
condition = condition.permute(0, 2, 1, 3, 4)  # (1, 24, 8, 64, 64)

# Sample from the model
with torch.no_grad():
    # This will generate the full video
    sampled_video = imagen.sample(
        batch_size=1,
        start_at_unet_number=1,
        stop_at_unet_number=1,
        cond_images=condition,  # Condition on past frames
        return_all_unet_outputs=False
    )

print(f"  Generated video shape: {sampled_video.shape}")
torch.save(sampled_video, OUTPUT_DIR / 'sample_prediction.pt')
print(f"  Saved sample to: {OUTPUT_DIR / 'sample_prediction.pt'}")

print("\n" + "=" * 80)
print("🎉 Video Diffusion Model Training Complete!")
print("=" * 80)

