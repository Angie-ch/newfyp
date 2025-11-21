"""
Manual DDPM training for the hybrid typhoon predictor.

This version:
1. Uses the Imagen-style VideoUNet3D from `custom_video_diffusion.py`
2. Adds the LT3P multi-task heads (structure, track, intensity, pressure)
3. Extends the dataset to emit multi-modal targets (track/intensity/pressure)
4. Introduces staged training: deterministic → diffusion → joint
5. Logs LT3P + diffusion metrics for regression tracking
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from custom_video_diffusion import VideoUNet3D
from hybrid_typhoon_predictor_v3 import ConvLSTM
from models.diffusion.prediction_heads import (
    IntensityHead,
    PressureHead,
    StructureHead,
    TrackHead,
)


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
            unet: VideoUNet3D model
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
        predicted_noise = unet(x_noisy, t, cond_video)
        
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
        predicted_noise = unet(x, t, cond_video)
        
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
# Dataset and Training
# ============================================================================

class HybridTyphoonPredictor_ManualDDPM(nn.Module):
    """
    ConvLSTM encoder + Imagen-style VideoUNet3D decoder + LT3P heads
    """

    def __init__(
        self,
        input_channels: int = 24,
        hidden_channels: int = 64,
        physics_channels: int = 128,
        output_channels: int = 24,
        past_timesteps: int = 8,
        future_timesteps: int = 12,
        image_size: tuple = (64, 64),
        diffusion_timesteps: int = 250,
        unet_dim: int = 32,
        unet_dim_mults: tuple = (1, 2, 4, 8),
        unet_num_resnet_blocks: int = 3,
    ):
        super().__init__()

        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        self.image_size = image_size
        self.output_channels = output_channels
        self.physics_channels = physics_channels

        # ConvLSTM encoder (temporal prior)
        self.convlstm_encoder = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=[hidden_channels, hidden_channels],
            kernel_size=3,
            num_layers=2,
            batch_first=True,
        )

        # Project ConvLSTM features into physics-aware channels
        self.cond_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, physics_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, physics_channels),
            nn.SiLU(),
            nn.Conv2d(physics_channels, physics_channels, kernel_size=1),
            nn.SiLU(),
        )

        # LT3P-style multi-task heads
        self.structure_head = StructureHead(physics_channels, output_channels)
        self.track_head = TrackHead(physics_channels, output_frames=future_timesteps)
        self.intensity_head = IntensityHead(physics_channels, output_frames=future_timesteps)
        self.pressure_head = PressureHead(physics_channels, output_frames=future_timesteps)

        # Imagen-style VideoUNet3D (fully 5D)
        self.video_unet = VideoUNet3D(
            in_channels=output_channels,
            out_channels=output_channels,
            cond_channels=physics_channels,
            base_channels=unet_dim,
            channel_mults=unet_dim_mults,
            num_res_blocks=unet_num_resnet_blocks,
        )

        # Manual diffusion process
        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_timesteps,
            beta_schedule='linear',
        )

        self.current_stage = 'joint'

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], flag: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = flag

    def configure_stage(self, stage_name: str):
        stage = stage_name.lower()
        if stage not in {'deterministic', 'diffusion', 'joint'}:
            raise ValueError(f"Unknown stage: {stage_name}")

        encoder_flag = stage in {'deterministic', 'joint'}
        heads_flag = stage in {'deterministic', 'joint'}
        diffusion_flag = stage in {'diffusion', 'joint'}

        self._set_requires_grad(self.convlstm_encoder, encoder_flag)
        self._set_requires_grad(self.cond_proj, encoder_flag)
        self._set_requires_grad(self.structure_head, heads_flag)
        self._set_requires_grad(self.track_head, heads_flag)
        self._set_requires_grad(self.intensity_head, heads_flag)
        self._set_requires_grad(self.pressure_head, heads_flag)
        self._set_requires_grad(self.video_unet, diffusion_flag)

        self.current_stage = stage

    def _encode_spatiotemporal(self, past_frames: torch.Tensor) -> torch.Tensor:
        _, last_state = self.convlstm_encoder(past_frames)
        conv_features = last_state[-1][0]  # (B, hidden_channels, H, W)
        return conv_features

    def _build_cond_video(self, conv_features: torch.Tensor) -> torch.Tensor:
        cond_spatial = self.cond_proj(conv_features)  # (B, physics_channels, H, W)
        cond_video = cond_spatial.unsqueeze(2).repeat(1, 1, self.future_timesteps, 1, 1)
        return cond_video

    def forward(self, past_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns multi-task predictions along with the conditioning video tensor.
        """
        conv_features = self._encode_spatiotemporal(past_frames)
        cond_video = self._build_cond_video(conv_features)

        structure_pred = self.structure_head(cond_video)  # (B, T, C, H, W)
        track_pred = self.track_head(cond_video)          # (B, T, 2)
        intensity_pred = self.intensity_head(cond_video)  # (B, T)
        pressure_pred = self.pressure_head(cond_video)    # (B, T)

        return {
            'cond_video': cond_video,
            'structure_pred': structure_pred,
            'track_pred': track_pred,
            'intensity_pred': intensity_pred,
            'pressure_pred': pressure_pred,
        }

    @torch.no_grad()
    def sample(self, past_frames: torch.Tensor, device: Optional[torch.device] = None):
        device = device or past_frames.device
        preds = self.forward(past_frames)
        cond_video = preds['cond_video']

        shape = (past_frames.shape[0], self.output_channels, self.future_timesteps, *self.image_size)
        sampled = self.diffusion.p_sample_loop(self.video_unet, shape, cond_video, device)
        sampled = sampled.permute(0, 2, 1, 3, 4)

        return {
            'future_frames': sampled,
            'track_pred': preds['track_pred'],
            'intensity_pred': preds['intensity_pred'],
            'pressure_pred': preds['pressure_pred'],
        }


class TyphoonDataset(Dataset):
    """
    Loads multi-modal ERA5 + IBTrACS samples from NPZ files.

    Returned dictionary keys:
        - past_frames:        (T_past, C, H, W)
        - future_frames:      (T_future, C, H, W)
        - track_past:         (T_past, 2)
        - track_future:       (T_future, 2)
        - intensity_past:     (T_past,)
        - intensity_future:   (T_future,)
        - pressure_past:      (T_past,)
        - pressure_future:    (T_future,)
        - storm_id:           string identifier
    """

    def __init__(self, data_dir: Path, normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        self.normalize = normalize

        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

        print(f"[INFO] Found {len(self.files)} samples in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        npz_path = self.files[idx]
        with np.load(npz_path) as data:
            past_frames = data['past_frames'].astype(np.float32)
            future_frames = data['future_frames'].astype(np.float32)
            storm_id = str(data['storm_id']) if 'storm_id' in data else npz_path.stem

            T_past = past_frames.shape[0]
            T_future = future_frames.shape[0]

            def _get_array(key: str, shape, dtype=np.float32):
                if key in data:
                    return data[key].astype(dtype)
                return np.zeros(shape, dtype=dtype)

            track_past = _get_array('track_past', (T_past, 2))
            track_future = _get_array('track_future', (T_future, 2))
            intensity_past = _get_array('past_intensity', (T_past,))
            intensity_future = _get_array('future_intensity', (T_future,))
            pressure_past = _get_array('past_pressure', (T_past,))
            pressure_future = _get_array('future_pressure', (T_future,))

        sample = {
            'past_frames': torch.from_numpy(past_frames),
            'future_frames': torch.from_numpy(future_frames),
            'track_past': torch.from_numpy(track_past),
            'track_future': torch.from_numpy(track_future),
            'intensity_past': torch.from_numpy(intensity_past),
            'intensity_future': torch.from_numpy(intensity_future),
            'pressure_past': torch.from_numpy(pressure_past),
            'pressure_future': torch.from_numpy(pressure_future),
            'storm_id': storm_id,
        }

        # Placeholder for future normalization hooks
        if self.normalize:
            pass

        return sample


def train_manual_ddpm(
    model: HybridTyphoonPredictor_ManualDDPM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-4,
    save_dir: str = 'checkpoints_manual_ddpm',
    deterministic_epochs: int = 10,
    diffusion_epochs: int = 20,
    joint_epochs: int = 20,
    lr_diffusion: Optional[float] = None,
    lr_joint: Optional[float] = None,
):
    """
    Three-phase training loop.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    model.diffusion.to(device)

    phases = [
        {
            'name': 'deterministic',
            'epochs': deterministic_epochs,
            'lambdas': {
                'structure': 1.0,
                'track': 1.0,
                'intensity': 0.5,
                'pressure': 0.5,
                'diffusion': 0.0,
            },
            'lr': lr,
        },
        {
            'name': 'diffusion',
            'epochs': diffusion_epochs,
            'lambdas': {
                'structure': 0.0,
                'track': 0.0,
                'intensity': 0.0,
                'pressure': 0.0,
                'diffusion': 1.0,
            },
            'lr': lr_diffusion or lr,
        },
        {
            'name': 'joint',
            'epochs': joint_epochs,
            'lambdas': {
                'structure': 1.0,
                'track': 1.0,
                'intensity': 0.5,
                'pressure': 0.5,
                'diffusion': 1.0,
            },
            'lr': lr_joint or lr,
        },
    ]

    history: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    best_val_loss = float('inf')

    print("\n" + "=" * 60)
    print("STARTING STAGED MANUAL DDPM TRAINING")
    print("=" * 60)

    for phase in phases:
        stage_name = phase['name']
        epochs = phase['epochs']
        lambdas = phase['lambdas']
        phase_lr = phase['lr']

        if epochs <= 0:
            continue

        print(f"\n>>> Stage: {stage_name.upper()} ({epochs} epochs)")
        model.configure_stage(stage_name)

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            print(f"[WARN] No trainable parameters for stage {stage_name}, skipping.")
            continue

        optimizer = torch.optim.Adam(params, lr=phase_lr)
        history[stage_name] = {'train': defaultdict(list), 'val': defaultdict(list)}

        for epoch in range(epochs):
            model.train()
            train_metrics = defaultdict(list)
            pbar = tqdm(train_loader, desc=f"Stage {stage_name} | Train {epoch+1}/{epochs}")

            for batch in pbar:
                past_frames = batch['past_frames'].to(device)
                future_frames = batch['future_frames'].to(device)
                track_future = batch['track_future'].to(device)
                intensity_future = batch['intensity_future'].to(device)
                pressure_future = batch['pressure_future'].to(device)

                preds = model(past_frames)
                cond_video = preds['cond_video']
                structure_pred = preds['structure_pred']
                track_pred = preds['track_pred']
                intensity_pred = preds['intensity_pred']
                pressure_pred = preds['pressure_pred']

                structure_loss = F.l1_loss(structure_pred, future_frames)
                track_loss = F.mse_loss(track_pred, track_future)
                intensity_loss = F.mse_loss(intensity_pred, intensity_future)
                pressure_loss = F.mse_loss(pressure_pred, pressure_future)

                if lambdas['diffusion'] > 0:
                    B = past_frames.shape[0]
                    t = torch.randint(0, model.diffusion.timesteps, (B,), device=device).long()
                    future_perm = future_frames.permute(0, 2, 1, 3, 4)
                    diffusion_loss = model.diffusion.p_losses(
                        model.video_unet,
                        future_perm,
                        t,
                        cond_video,
                    )
                else:
                    diffusion_loss = torch.zeros(1, device=device)

                total_loss = (
                    lambdas['structure'] * structure_loss
                    + lambdas['track'] * track_loss
                    + lambdas['intensity'] * intensity_loss
                    + lambdas['pressure'] * pressure_loss
                    + lambdas['diffusion'] * diffusion_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                metrics = {
                    'structure_loss': structure_loss.item(),
                    'track_loss': track_loss.item(),
                    'track_mae': F.l1_loss(track_pred, track_future).item(),
                    'intensity_loss': intensity_loss.item(),
                    'intensity_mae': F.l1_loss(intensity_pred, intensity_future).item(),
                    'pressure_loss': pressure_loss.item(),
                    'pressure_mae': F.l1_loss(pressure_pred, pressure_future).item(),
                    'diffusion_loss': diffusion_loss.item(),
                    'total_loss': total_loss.item(),
                }

                for key, value in metrics.items():
                    train_metrics[key].append(value)

                pbar.set_postfix({
                    'tot': f"{metrics['total_loss']:.4f}",
                    'diff': f"{metrics['diffusion_loss']:.4f}",
                    'trk': f"{metrics['track_mae']:.4f}",
                    'int': f"{metrics['intensity_mae']:.4f}",
                })

            avg_train = {k: float(np.mean(v)) for k, v in train_metrics.items()}
            for key, value in avg_train.items():
                history[stage_name]['train'][key].append(value)

            # Validation
            model.eval()
            val_metrics = defaultdict(list)
            with torch.no_grad():
                for batch in val_loader:
                    past_frames = batch['past_frames'].to(device)
                    future_frames = batch['future_frames'].to(device)
                    track_future = batch['track_future'].to(device)
                    intensity_future = batch['intensity_future'].to(device)
                    pressure_future = batch['pressure_future'].to(device)

                    preds = model(past_frames)
                    cond_video = preds['cond_video']
                    structure_pred = preds['structure_pred']
                    track_pred = preds['track_pred']
                    intensity_pred = preds['intensity_pred']
                    pressure_pred = preds['pressure_pred']

                    structure_loss = F.l1_loss(structure_pred, future_frames)
                    track_loss = F.mse_loss(track_pred, track_future)
                    intensity_loss = F.mse_loss(intensity_pred, intensity_future)
                    pressure_loss = F.mse_loss(pressure_pred, pressure_future)

                    if lambdas['diffusion'] > 0:
                        B = past_frames.shape[0]
                        t = torch.randint(0, model.diffusion.timesteps, (B,), device=device).long()
                        future_perm = future_frames.permute(0, 2, 1, 3, 4)
                        diffusion_loss = model.diffusion.p_losses(
                            model.video_unet,
                            future_perm,
                            t,
                            cond_video,
                        )
                    else:
                        diffusion_loss = torch.zeros(1, device=device)

                    total_loss = (
                        lambdas['structure'] * structure_loss
                        + lambdas['track'] * track_loss
                        + lambdas['intensity'] * intensity_loss
                        + lambdas['pressure'] * pressure_loss
                        + lambdas['diffusion'] * diffusion_loss
                    )

                    metrics = {
                        'structure_loss': structure_loss.item(),
                        'track_loss': track_loss.item(),
                        'track_mae': F.l1_loss(track_pred, track_future).item(),
                        'intensity_loss': intensity_loss.item(),
                        'intensity_mae': F.l1_loss(intensity_pred, intensity_future).item(),
                        'pressure_loss': pressure_loss.item(),
                        'pressure_mae': F.l1_loss(pressure_pred, pressure_future).item(),
                        'diffusion_loss': diffusion_loss.item(),
                        'total_loss': total_loss.item(),
                    }
                    for key, value in metrics.items():
                        val_metrics[key].append(value)

            avg_val = {k: float(np.mean(v)) for k, v in val_metrics.items()}
            for key, value in avg_val.items():
                history[stage_name]['val'][key].append(value)

            print(
                f"\nStage {stage_name} | Epoch {epoch+1}/{epochs} "
                f"- Train Total: {avg_train['total_loss']:.4f}, "
                f"Val Total: {avg_val['total_loss']:.4f}, "
                f"Val Track MAE: {avg_val['track_mae']:.4f}, "
                f"Val Int MAE: {avg_val['intensity_mae']:.4f}"
            )

            current_val_total = avg_val['total_loss']
            if current_val_total < best_val_loss:
                best_val_loss = current_val_total
                checkpoint_path = save_dir / 'best_model.pth'
                torch.save(
                    {
                        'stage': stage_name,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'history': history,
                    },
                    checkpoint_path,
                )
                print(f"  [OK] Saved best model (val_total={best_val_loss:.4f})")

            with open(save_dir / 'staged_training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print(f"Best validation total loss: {best_val_loss:.4f}")
    print("=" * 60)

    return history


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {DEVICE}")
    
    # Data directories
    DATA_DIR = Path('/Users/angiecheong/Desktop/fyp3/data/data/era5/typhoon_data_2018_2021_full')
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
        physics_channels=128,
        output_channels=24,
        past_timesteps=8,
        future_timesteps=12,
        image_size=(64, 64),
        diffusion_timesteps=DIFFUSION_TIMESTEPS,
    ).to(DEVICE)
    
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
        lr=LEARNING_RATE,
        save_dir='checkpoints_manual_ddpm',
        deterministic_epochs=10,
        diffusion_epochs=20,
        joint_epochs=20,
    )
    
    print("\n[SUCCESS] All done! Check 'checkpoints_manual_ddpm/' for results.")

