"""
Manual DDPM Training for Hybrid Typhoon Predictor
Video-to-Video Diffusion Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math

# Import our hybrid model (we'll use the ConvLSTM + Unet3D parts)
from hybrid_typhoon_predictor_v3 import (
    ConvLSTM, TrackPredictor
)
from imagen_pytorch import Unet3D


# ============================================================================
# DDPM Utilities: Noise Schedule & Diffusion Process
# ============================================================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule for beta (variance schedule)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    """
    Manual DDPM implementation for video-to-video diffusion
    """
    def __init__(self, timesteps=250, beta_schedule='linear'):
        self.timesteps = timesteps
        
        # Define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Pre-compute useful quantities
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def to(self, device):
        """Move all tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to x_start at timestep t
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_start: (B, C, T, H, W) - clean video
            t: (B,) - timesteps
            noise: (B, C, T, H, W) - noise to add
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Extract coefficients at timestep t
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Add noise: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
    
    def _extract(self, a, t, x_shape):
        """
        Extract coefficients at timestep t and reshape for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        # Reshape to (B, 1, 1, 1, 1) for 5D video tensors
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def p_losses(self, unet, x_start, t, cond_video, noise=None):
        """
        Training loss: predict noise at timestep t
        
        Args:
            unet: Unet3D model
            x_start: (B, C, T, H, W) - clean future frames
            t: (B,) - timesteps
            cond_video: (B, C_cond, T, H, W) - condition from ConvLSTM
            noise: (B, C, T, H, W) - noise to add
        
        Returns:
            loss: scalar tensor
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to x_start → x_t
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise using Unet3D
        # Unet3D expects: (x, time, cond_video_frames) for 5D video tensors
        predicted_noise = unet(
            x_noisy,
            time=t,
            cond_video_frames=cond_video
        )
        
        # Compute MSE loss between predicted and true noise
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, unet, x, t, cond_video):
        """
        Reverse diffusion: denoise x_t → x_{t-1}
        
        Args:
            unet: Unet3D model
            x: (B, C, T, H, W) - noisy video at timestep t
            t: (B,) - current timesteps
            cond_video: (B, C_cond, T, H, W) - condition
        
        Returns:
            x_{t-1}: (B, C, T, H, W)
        """
        # Predict noise
        predicted_noise = unet(x, time=t, cond_video_frames=cond_video)
        
        # Extract coefficients
        alpha = self._extract(self.alphas, t, x.shape)
        alpha_cumprod = self._extract(self.alphas_cumprod, t, x.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        
        # Predict x_0 from x_t and predicted noise
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod * predicted_noise) / torch.sqrt(alpha_cumprod)
        
        # Clip x_0 to [-1, 1] (data range)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute x_{t-1}
        alpha_prev = self._extract(self.alphas_cumprod_prev, t, x.shape)
        dir_x_t = torch.sqrt(1.0 - alpha_prev) * predicted_noise
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_x_t
        
        # Add noise if not the last step
        if t[0] > 0:
            noise = torch.randn_like(x)
            variance = self._extract(self.posterior_variance, t, x.shape)
            x_prev = x_prev + torch.sqrt(variance) * noise
        
        return x_prev
    
    @torch.no_grad()
    def p_sample_loop(self, unet, shape, cond_video, device):
        """
        Full sampling loop: x_T → x_{T-1} → ... → x_0
        
        Args:
            unet: Unet3D model
            shape: (B, C, T, H, W) - desired output shape
            cond_video: (B, C_cond, T, H, W) - condition
            device: torch device
        
        Returns:
            x_0: (B, C, T, H, W) - generated clean video
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Progressively denoise
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(unet, x, t, cond_video)
        
        return x


# ============================================================================
# Hybrid Model with Manual DDPM
# ============================================================================

class HybridTyphoonPredictor_ManualDDPM(nn.Module):
    """
    Hybrid model: ConvLSTM encoder + Manual DDPM decoder + Track predictor
    """
    def __init__(self, input_channels=24, hidden_channels=64, output_channels=24,
                 past_timesteps=8, future_timesteps=12, image_size=(64, 64),
                 diffusion_timesteps=250):
        super().__init__()
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        self.image_size = image_size
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        # ConvLSTM Encoder (from Model A)
        self.convlstm_encoder = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=[hidden_channels, hidden_channels],
            kernel_size=3,
            num_layers=2,
            batch_first=True
        )
        
        # Condition projection: ConvLSTM features → Unet3D condition
        # Input: (B, C_hidden, H, W) → expand to (B, C_hidden, T_future, H, W)
        # Output: (B, C_out, T_future, H, W)
        self.cond_proj = nn.Conv3d(hidden_channels, output_channels, kernel_size=1)
        
        # Track Predictor (MLP)
        self.track_predictor = TrackPredictor(
            hidden_dim=hidden_channels * image_size[0] * image_size[1],
            output_frames=future_timesteps
        )
        
        # Unet3D for diffusion (Model B)
        print("[INFO] Creating Unet3D for manual DDPM...")
        self.video_unet = Unet3D(
            dim=32,  # Base dimension
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=3,
            channels=output_channels,  # ERA5 channels
            # For video conditioning, don't use cond_video_frames directly in forward()
            # IMPORTANT: Disable ALL attention to avoid 5D/4D issues
            layer_attns=(False, False, False, False),
            use_linear_attn=False,  # Disable linear attention too
            memory_efficient=True,
            init_conv_to_final_conv_residual=True,
        )
        print("[OK] Unet3D initialized (NO ATTENTION)")
        
        # Gaussian Diffusion process
        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_timesteps,
            beta_schedule='linear'
        )
        print(f"[OK] Gaussian Diffusion initialized with {diffusion_timesteps} timesteps")
    
    def encode_past_frames(self, past_frames):
        """
        Encode past frames using ConvLSTM
        
        Args:
            past_frames: (B, T_past, C_in, H, W)
        
        Returns:
            convlstm_features: (B, C_hidden, H, W)
            cond_video: (B, C_out, T_future, H, W)
            predicted_track: (B, T_future, 2)
        """
        B, T_past, C_in, H, W = past_frames.shape
        
        # 1. ConvLSTM Encoder
        _, last_state = self.convlstm_encoder(past_frames)
        convlstm_features = last_state[-1][0]  # (B, C_hidden, H, W)
        
        # 2. Condition Projection
        # Expand to (B, C_hidden, T_future, H, W)
        cond_video_expanded = convlstm_features.unsqueeze(2).repeat(1, 1, self.future_timesteps, 1, 1)
        # Project to (B, C_out, T_future, H, W)
        cond_video = self.cond_proj(cond_video_expanded)
        
        # 3. Track Prediction
        track_input = convlstm_features.reshape(B, -1)
        predicted_track = self.track_predictor(track_input)
        
        return convlstm_features, cond_video, predicted_track
    
    def forward(self, past_frames, future_frames, track_future, t=None):
        """
        Training forward pass
        
        Args:
            past_frames: (B, T_past, C, H, W)
            future_frames: (B, T_future, C, H, W)
            track_future: (B, T_future, 2)
            t: (B,) - timesteps for diffusion (if None, randomly sampled)
        
        Returns:
            diffusion_loss: scalar
            track_loss: scalar
            predicted_track: (B, T_future, 2)
        """
        B = past_frames.shape[0]
        device = past_frames.device
        
        # Encode past frames
        _, cond_video, predicted_track = self.encode_past_frames(past_frames)
        
        # Track loss
        track_loss = F.mse_loss(predicted_track, track_future)
        
        # Diffusion loss
        # Convert future_frames from (B, T, C, H, W) to (B, C, T, H, W)
        future_frames_perm = future_frames.permute(0, 2, 1, 3, 4)
        
        # Sample timesteps if not provided
        if t is None:
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=device).long()
        
        # Compute diffusion loss using manual DDPM
        diffusion_loss = self.diffusion.p_losses(
            self.video_unet,
            future_frames_perm,
            t,
            cond_video
        )
        
        return diffusion_loss, track_loss, predicted_track
    
    @torch.no_grad()
    def sample(self, past_frames):
        """
        Inference: generate future frames
        
        Args:
            past_frames: (B, T_past, C, H, W)
        
        Returns:
            sampled_frames: (B, T_future, C, H, W)
            predicted_track: (B, T_future, 2)
        """
        B = past_frames.shape[0]
        device = past_frames.device
        
        # Encode past frames
        _, cond_video, predicted_track = self.encode_past_frames(past_frames)
        
        # Sample future frames using DDPM
        shape = (B, self.output_channels, self.future_timesteps, *self.image_size)
        sampled_frames_perm = self.diffusion.p_sample_loop(
            self.video_unet,
            shape,
            cond_video,
            device
        )
        
        # Convert back to (B, T, C, H, W)
        sampled_frames = sampled_frames_perm.permute(0, 2, 1, 3, 4)
        
        return sampled_frames, predicted_track


# ============================================================================
# Dataset and Training
# ============================================================================

class TyphoonDataset(Dataset):
    def __init__(self, data_dir, normalize=True):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        self.normalize = normalize
        
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        print(f"[INFO] Found {len(self.files)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        past_frames = torch.from_numpy(data['past_frames']).float()
        future_frames = torch.from_numpy(data['future_frames']).float()
        track_past = torch.from_numpy(data['track_past']).float()
        track_future = torch.from_numpy(data['track_future']).float()
        
        # Normalize to [-1, 1] for diffusion
        if self.normalize:
            # ERA5 data ranges (approximate)
            # We'll use simple standardization: (x - mean) / std
            # For now, just scale to [-1, 1] assuming data is already somewhat normalized
            pass  # Data is already normalized in preprocessing
        
        return past_frames, future_frames, track_past, track_future


def train_manual_ddpm(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    lr=1e-4,
    save_dir='checkpoints_manual_ddpm',
    lambda_track=1.0,
    lambda_diffusion=1.0
):
    """
    Training loop for manual DDPM
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Move diffusion schedule to device
    model.diffusion.to(device)
    
    # Training history
    history = {
        'train_diffusion_loss': [],
        'train_track_loss': [],
        'train_total_loss': [],
        'val_diffusion_loss': [],
        'val_track_loss': [],
        'val_total_loss': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("STARTING MANUAL DDPM TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # ==================== Training ====================
        model.train()
        train_diffusion_losses = []
        train_track_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (past_frames, future_frames, track_past, track_future) in enumerate(pbar):
            # Move to device
            past_frames = past_frames.to(device)
            future_frames = future_frames.to(device)
            track_future = track_future.to(device)
            
            # Forward pass
            diffusion_loss, track_loss, _ = model(
                past_frames, future_frames, track_future
            )
            
            # Total loss
            total_loss = lambda_diffusion * diffusion_loss + lambda_track * track_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Log losses
            train_diffusion_losses.append(diffusion_loss.item())
            train_track_losses.append(track_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{diffusion_loss.item():.4f}',
                'T_loss': f'{track_loss.item():.4f}',
                'Total': f'{total_loss.item():.4f}'
            })
        
        # Average training losses
        avg_train_diffusion = np.mean(train_diffusion_losses)
        avg_train_track = np.mean(train_track_losses)
        avg_train_total = avg_train_diffusion * lambda_diffusion + avg_train_track * lambda_track
        
        history['train_diffusion_loss'].append(avg_train_diffusion)
        history['train_track_loss'].append(avg_train_track)
        history['train_total_loss'].append(avg_train_total)
        
        # ==================== Validation ====================
        model.eval()
        val_diffusion_losses = []
        val_track_losses = []
        
        with torch.no_grad():
            for past_frames, future_frames, track_past, track_future in val_loader:
                past_frames = past_frames.to(device)
                future_frames = future_frames.to(device)
                track_future = track_future.to(device)
                
                diffusion_loss, track_loss, _ = model(
                    past_frames, future_frames, track_future
                )
                
                val_diffusion_losses.append(diffusion_loss.item())
                val_track_losses.append(track_loss.item())
        
        # Average validation losses
        avg_val_diffusion = np.mean(val_diffusion_losses)
        avg_val_track = np.mean(val_track_losses)
        avg_val_total = avg_val_diffusion * lambda_diffusion + avg_val_track * lambda_track
        
        history['val_diffusion_loss'].append(avg_val_diffusion)
        history['val_track_loss'].append(avg_val_track)
        history['val_total_loss'].append(avg_val_total)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - Diffusion: {avg_train_diffusion:.4f}, Track: {avg_train_track:.4f}, Total: {avg_train_total:.4f}")
        print(f"  Val   - Diffusion: {avg_val_diffusion:.4f}, Track: {avg_val_track:.4f}, Total: {avg_val_total:.4f}")
        
        # Save checkpoint
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'history': history
            }, checkpoint_path)
            print(f"  [OK] Saved best model (val_loss={best_val_loss:.4f})")
        
        # Save history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    return history


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {DEVICE}")
    
    # Data directories
    DATA_DIR = r'D:\typhoon_data_2018_2021_full'
    TRAIN_DIR = Path(DATA_DIR) / 'train' / 'cases'
    VAL_DIR = Path(DATA_DIR) / 'val' / 'cases'
    
    # Hyperparameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DIFFUSION_TIMESTEPS = 250
    
    # Create datasets
    print("\n[INFO] Loading datasets...")
    train_dataset = TyphoonDataset(TRAIN_DIR)
    val_dataset = TyphoonDataset(VAL_DIR)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[OK] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n[INFO] Creating model...")
    model = HybridTyphoonPredictor_ManualDDPM(
        input_channels=24,
        hidden_channels=64,
        output_channels=24,
        past_timesteps=8,
        future_timesteps=12,
        image_size=(64, 64),
        diffusion_timesteps=DIFFUSION_TIMESTEPS
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n[INFO] Starting training...")
    history = train_manual_ddpm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        save_dir='checkpoints_manual_ddpm',
        lambda_track=1.0,
        lambda_diffusion=1.0
    )
    
    print("\n[SUCCESS] All done! Check 'checkpoints_manual_ddpm/' for results.")

