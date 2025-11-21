"""
Hybrid Model: ConvLSTM Encoder + Video-to-Video Diffusion Decoder
Combines the best of both worlds:
- ConvLSTM: Efficient feature extraction from past frames
- Video Diffusion: High-quality temporal coherent generation

Inspired by LT3P: Uses diffusion for trajectory prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys

# Add forecast-video-diffmodels to path
sys.path.append('forecast-video-diffmodels/imagen')

try:
    from imagen_pytorch import Unet3D, Imagen, ImagenTrainer
    print("[OK] Successfully imported imagen_pytorch")
except ImportError as e:
    print(f"[ERROR] Failed to import: {e}")
    print("Please install: pip install imagen-pytorch einops")
    exit(1)

print("=" * 80)
print("HYBRID MODEL: ConvLSTM + Video-to-Video Diffusion")
print("=" * 80)

# Configuration
DATA_DIR = Path("D:/typhoon_data_2018_2021_full")
OUTPUT_DIR = Path("hybrid_model_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")

# ============================================================================
# PART 1: ConvLSTM Encoder (Feature Extractor)
# ============================================================================

class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell for processing spatial-temporal features"""
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
    
    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTMEncoder(nn.Module):
    """
    ConvLSTM Encoder: Extracts rich features from past frames
    Output: Condition features for Video Diffusion
    """
    def __init__(self, input_channels=24, hidden_channels=64, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Initial feature extraction
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        
        # Multi-layer ConvLSTM
        self.lstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Output projection for diffusion conditioning
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
    
    def forward(self, past_frames):
        """
        Args:
            past_frames: (B, T_past=8, C=24, H=64, W=64)
        Returns:
            condition_features: (B, T_past, C_hidden=64, H, W)
            final_hidden: (B, C_hidden, H, W) - for initial state
        """
        B, T, C, H, W = past_frames.shape
        
        # Initialize hidden states for each layer
        h_states = [torch.zeros(B, self.hidden_channels, H, W, device=past_frames.device)
                   for _ in self.lstm_layers]
        c_states = [torch.zeros(B, self.hidden_channels, H, W, device=past_frames.device)
                   for _ in self.lstm_layers]
        
        condition_features = []
        
        # Process each timestep
        for t in range(T):
            x_t = past_frames[:, t]  # (B, C, H, W)
            
            # Initial feature extraction
            x_t = self.input_conv(x_t)  # (B, hidden_channels, H, W)
            
            # Pass through ConvLSTM layers
            for layer_idx, lstm_layer in enumerate(self.lstm_layers):
                h_states[layer_idx], c_states[layer_idx] = lstm_layer(
                    x_t, (h_states[layer_idx], c_states[layer_idx])
                )
                x_t = h_states[layer_idx]
            
            # Project to condition features
            cond_t = self.output_proj(x_t)  # (B, hidden_channels, H, W)
            condition_features.append(cond_t)
        
        # Stack condition features
        condition_features = torch.stack(condition_features, dim=1)  # (B, T_past, C_hidden, H, W)
        
        # Final hidden state (from last layer, last timestep)
        final_hidden = h_states[-1]  # (B, C_hidden, H, W)
        
        return condition_features, final_hidden


# ============================================================================
# PART 2: Video-to-Video Diffusion Decoder
# ============================================================================

class VideoToVideoDiffusion(nn.Module):
    """
    Video-to-Video Diffusion Model using Imagen framework
    Conditions on past frames (from ConvLSTM) to generate future frames
    """
    def __init__(self, condition_channels=64, output_channels=24, hidden_dim=128):
        super().__init__()
        
        # Projection layer to map ConvLSTM features to condition embedding
        # Average pool over time, then flatten spatial dimensions
        # Input: (B, T_past, C_hidden, H, W) -> average -> (B, C_hidden, H, W) -> flatten -> (B, C_hidden*H*W)
        # Output: (B, condition_embed_dim)
        self.condition_embed_dim = 1024  # Match forecast-video-diffmodels pattern
        
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_channels * 64 * 64, 2048),  # First reduce
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, self.condition_embed_dim),  # Final embedding
        )
        
        # Projection layer to map ConvLSTM features to match output channels for cond_video_frames
        # Input: (B, C_hidden, T, H, W) -> Output: (B, C_out, T, H, W)
        self.condition_video_proj = nn.Sequential(
            nn.Conv3d(condition_channels, output_channels, kernel_size=1),  # Channel projection
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
        )
        
        # Create Unet3D for video diffusion
        # Based on forecast-video-diffmodels/v_m01w_woERA5_64_FC.py
        # Key finding: imagen-pytorch 2.1.0 DOES support video (tested successfully)
        # Use exact config from forecast-video-diffmodels
        self.unet = Unet3D(
            dim=32,  # Match forecast-video-diffmodels (not 128!)
            dim_mults=(1, 2, 4, 8),  # Channel multipliers
            channels=output_channels,  # Input/output channels (ERA5 variables)
            cond_dim=self.condition_embed_dim,  # Condition embedding dimension
            num_resnet_blocks=3,  # Match forecast-video-diffmodels
            layer_attns=(False, True, True, True),  # Attention at deeper layers (matches forecast-video-diffmodels)
            attn_heads=8,
            attn_dim_head=32,
            use_linear_attn=True,  # Use linear attention for efficiency
            memory_efficient=True,
            init_conv_to_final_conv_residual=True,
        )
        
        # Create Imagen for diffusion process
        # Note: Current imagen-pytorch version doesn't support condition_on_continuous
        # We'll handle conditioning directly in Unet3D via cond_video_frames
        # Disable text conditioning to avoid text_embeds issues
        self.imagen = Imagen(
            unets=[self.unet],  # List of unets (we only use one)
            image_sizes=64,  # Spatial resolution
            timesteps=250,  # Diffusion timesteps (matching forecast-video-diffmodels)
            channels=output_channels,
            cond_drop_prob=0.1,  # Condition dropout probability
            condition_on_text=False,  # Disable text conditioning
        )
        
        # Create trainer (handles training loop)
        self.trainer = None  # Will be created in training function
    
    def get_condition_embed(self, condition_features):
        """
        Convert condition features to embedding
        Args:
            condition_features: (B, T_past, C_hidden, H, W)
        Returns:
            condition_embed: (B, condition_embed_dim)
        """
        B = condition_features.shape[0]
        
        # Average pool over time dimension to get (B, C_hidden, H, W)
        condition_pooled = condition_features.mean(dim=1)  # (B, C_hidden, H, W)
        
        # Flatten spatial dimensions
        condition_flat = condition_pooled.reshape(B, -1)  # (B, C_hidden * H * W)
        
        # Project to embedding dimension
        condition_embed = self.condition_proj(condition_flat)  # (B, condition_embed_dim)
        
        return condition_embed
    
    def forward(self, noisy_future, condition_features, timestep):
        """
        Args:
            noisy_future: (B, T_future=12, C=24, H, W) - noisy future frames
            condition_features: (B, T_past=8, C_hidden=64, H, W) - from ConvLSTM
            timestep: (B,) - diffusion timestep
        Returns:
            predicted_noise: (B, T_future, C, H, W)
        """
        # Rearrange for Unet3D: (B, C, T, H, W)
        noisy_future = noisy_future.permute(0, 2, 1, 3, 4)  # (B, C=24, T=12, H, W)
        
        # Project condition features to embedding dimension
        # condition_features: (B, T_past, C_hidden, H, W)
        # Option 1: Use final hidden state (more efficient)
        # Option 2: Average pool over time, then flatten
        B = condition_features.shape[0]
        
        # Average pool over time dimension to get (B, C_hidden, H, W)
        condition_pooled = condition_features.mean(dim=1)  # (B, C_hidden, H, W)
        
        # Flatten spatial dimensions
        condition_flat = condition_pooled.reshape(B, -1)  # (B, C_hidden * H * W)
        
        # Project to embedding dimension
        condition_embed = self.condition_proj(condition_flat)  # (B, condition_embed_dim)
        
        # Unet3D expects condition as cond parameter
        # Rearrange noisy_future for Unet3D: (B, C, T, H, W)
        noisy_future_perm = noisy_future.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Unet3D forward signature: (x, time, cond=None, ...)
        # time should be (B,) tensor
        predicted_noise = self.unet(
            noisy_future_perm,
            time=timestep,  # Use 'time' instead of 'timestep'
            cond=condition_embed,  # Condition embedding
        )
        
        # Rearrange back: (B, T, C, H, W)
        predicted_noise = predicted_noise.permute(0, 2, 1, 3, 4)
        
        # Rearrange back: (B, T, C, H, W)
        predicted_noise = predicted_noise.permute(0, 2, 1, 3, 4)
        
        return predicted_noise


# ============================================================================
# PART 3: Hybrid Model (ConvLSTM + Video Diffusion)
# ============================================================================

class HybridTyphoonPredictor(nn.Module):
    """
    Hybrid Model combining ConvLSTM and Video Diffusion
    
    Architecture:
    1. ConvLSTM Encoder: Extract features from past frames
    2. Video Diffusion: Generate future frames conditioned on past features
    3. Track Predictor: Predict trajectory from features
    """
    def __init__(self, 
                 input_channels=24,
                 hidden_channels=64,
                 output_channels=24,
                 past_timesteps=8,
                 future_timesteps=12):
        super().__init__()
        
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        
        # ConvLSTM Encoder
        self.encoder = ConvLSTMEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=2
        )
        
        # Video-to-Video Diffusion Decoder
        self.diffusion = VideoToVideoDiffusion(
            condition_channels=hidden_channels,
            output_channels=output_channels,
            hidden_dim=hidden_channels
        )
        
        # Track Predictor (from ConvLSTM features)
        self.track_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, future_timesteps * 2),  # (lon, lat) for each timestep
        )
    
    def encode(self, past_frames):
        """Encode past frames using ConvLSTM"""
        condition_features, final_hidden = self.encoder(past_frames)
        return condition_features, final_hidden
    
    def predict_track(self, final_hidden):
        """Predict trajectory from encoded features"""
        track_flat = self.track_predictor(final_hidden)  # (B, future_timesteps * 2)
        track = track_flat.reshape(-1, self.future_timesteps, 2)  # (B, T_future, 2)
        return track
    
    def forward(self, past_frames, future_frames=None, timestep=None):
        """
        Training: Predict noise for diffusion
        Inference: Generate future frames
        
        Args:
            past_frames: (B, T_past=8, C=24, H, W)
            future_frames: (B, T_future=12, C=24, H, W) - only for training
            timestep: (B,) - only for training
        """
        # Encode past frames
        condition_features, final_hidden = self.encode(past_frames)
        
        # Predict track
        predicted_track = self.predict_track(final_hidden)
        
        # Return condition features (will be used as cond_video_frames)
        # and predicted track
        return condition_features, final_hidden, predicted_track


# ============================================================================
# PART 4: Training Dataset
# ============================================================================

class HybridDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / split / 'cases'
        self.samples = sorted(list(self.data_dir.glob('*.npz')))
        print(f"  Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        
        past_frames = torch.FloatTensor(data['past_frames'])    # (8, 24, 64, 64)
        future_frames = torch.FloatTensor(data['future_frames']) # (12, 24, 64, 64)
        track_past = torch.FloatTensor(data['track_past'])       # (8, 2)
        track_future = torch.FloatTensor(data['track_future'])   # (12, 2)
        
        return {
            'past_frames': past_frames,
            'future_frames': future_frames,
            'track_past': track_past,
            'track_future': track_future,
        }


# ============================================================================
# PART 5: Training Function
# ============================================================================

def train_hybrid_model():
    """Train the hybrid ConvLSTM + Video Diffusion model"""
    
    print("\n[1] Loading datasets...")
    train_dataset = HybridDataset(DATA_DIR, 'train')
    val_dataset = HybridDataset(DATA_DIR, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n[2] Creating hybrid model...")
    model = HybridTyphoonPredictor(
        input_channels=24,
        hidden_channels=64,
        output_channels=24,
        past_timesteps=8,
        future_timesteps=12
    ).to(DEVICE)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create ImagenTrainer (handles diffusion training)
    trainer = ImagenTrainer(
        model.diffusion.imagen,
        lr=LEARNING_RATE,
        use_ema=True,  # Use exponential moving average
        verbose=False
    ).to(DEVICE)
    
    # Track predictor optimizer (separate from diffusion)
    track_optimizer = torch.optim.Adam(model.track_predictor.parameters(), lr=LEARNING_RATE)
    
    # Loss function for track
    track_loss_fn = nn.MSELoss()
    
    print("\n[3] Training hybrid model...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_track_loss': [], 'val_track_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training
        model.train()
        train_loss_total = 0
        train_track_loss_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            past_frames = batch['past_frames'].to(DEVICE)
            future_frames = batch['future_frames'].to(DEVICE)  # (B, T_future=12, C=24, H, W)
            track_future = batch['track_future'].to(DEVICE)
            
            # Forward pass through model
            # Returns: condition_features, final_hidden, predicted_track
            condition_features, final_hidden, predicted_track = model(
                past_frames,
                future_frames=future_frames
            )
            
            # Rearrange future_frames for ImagenTrainer: (B, T, C, H, W) -> (B, C, T, H, W)
            future_frames_perm = future_frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Prepare condition video: (B, T_past, C_hidden, H, W) -> (B, C_out, T_past, H, W)
            condition_video = condition_features.permute(0, 2, 1, 3, 4)  # (B, C_hidden, T_past, H, W)
            condition_video = model.diffusion.condition_video_proj(condition_video)  # (B, C_out, T_past, H, W)
            
            # Use ImagenTrainer for training (handles diffusion internally)
            # This matches forecast-video-diffmodels pattern
            # Since condition_on_text=False, we don't need to pass text_embeds
            diffusion_loss = trainer(
                future_frames_perm,  # Target video: (B, C, T, H, W)
                cond_video_frames=condition_video,  # Condition video: (B, C, T_cond, H, W)
                unet_number=1,
                ignore_time=False  # Use temporal information
            )
            
            # Update diffusion model
            trainer.update(unet_number=1)
            
            # Train track predictor separately
            track_optimizer.zero_grad()
            track_loss = track_loss_fn(predicted_track, track_future)
            track_loss.backward()
            track_optimizer.step()
            
            train_loss_total += diffusion_loss.item()
            train_track_loss_total += track_loss.item()
            
            pbar.set_postfix({
                'diff_loss': f'{diffusion_loss.item():.4f}',
                'track_loss': f'{track_loss.item():.4f}'
            })
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_track = train_track_loss_total / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_total = 0
        val_track_loss_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                past_frames = batch['past_frames'].to(DEVICE)
                future_frames = batch['future_frames'].to(DEVICE)
                track_future = batch['track_future'].to(DEVICE)
                
                condition_features, final_hidden, predicted_track = model(
                    past_frames,
                    future_frames=future_frames
                )
                
                future_frames_perm = future_frames.permute(0, 2, 1, 3, 4)
                
                # Prepare condition video
                condition_video = condition_features.permute(0, 2, 1, 3, 4)
                condition_video = model.diffusion.condition_video_proj(condition_video)
                
                # Validation loss using trainer (no update)
                diffusion_loss = trainer(
                    future_frames_perm,
                    cond_video_frames=condition_video,
                    unet_number=1,
                    ignore_time=False
                )
                
                track_loss = track_loss_fn(predicted_track, track_future)
                
                val_loss_total += diffusion_loss.item()
                val_track_loss_total += track_loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_track = val_track_loss_total / len(val_loader)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_track_loss'].append(avg_train_track)
        history['val_track_loss'].append(avg_val_track)
        
        print(f"  Train - Diffusion: {avg_train_loss:.4f}, Track: {avg_train_track:.4f}")
        print(f"  Val   - Diffusion: {avg_val_loss:.4f}, Track: {avg_val_track:.4f}")
        
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save model state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'track_optimizer_state_dict': track_optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, OUTPUT_DIR / 'best_hybrid_model.pt')
        # Save trainer state (includes diffusion model and EMA)
        trainer.save(OUTPUT_DIR / 'best_hybrid_trainer.pt')
        print(f"  [BEST] Saved model (val_loss: {avg_val_loss:.4f})")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'track_optimizer_state_dict': track_optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pt')
        trainer.save(OUTPUT_DIR / f'checkpoint_trainer_epoch_{epoch+1}.pt')
        print(f"  [CHECKPOINT] Saved at epoch {epoch+1}")
    
    # Save history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'best_hybrid_model.pt'}")
    
    return model


if __name__ == '__main__':
    model = train_hybrid_model()

