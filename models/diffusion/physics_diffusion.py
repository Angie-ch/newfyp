"""
Physics-Informed Multi-Task Diffusion Model

Main model combining all three innovations:
1. Physics constraints
2. Typhoon-aware architecture
3. Multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import TyphoonAwareUNet3D
from .prediction_heads import StructureHead, TrackHead, IntensityHead
from .physics_constraints import PhysicsProjector


class PhysicsInformedDiffusionModel(nn.Module):
    """
    Complete physics-informed multi-task diffusion model
    
    Combines:
    - Typhoon-aware UNet backbone
    - Multi-task prediction heads
    - Physics constraint layers
    """
    
    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        depth: int = 3,
        output_frames: int = 8,
        use_physics_projector: bool = True,
        use_spiral_attention: bool = True,
        use_multiscale_temporal: bool = True
    ):
        """
        Initialize model
        
        Args:
            latent_channels: Number of latent channels
            hidden_dim: Hidden dimension for UNet
            num_heads: Number of attention heads
            depth: UNet depth
            output_frames: Number of output timesteps
            use_physics_projector: Enable physics projection
            use_spiral_attention: Enable spiral attention
            use_multiscale_temporal: Enable multi-scale temporal modeling
        """
        super().__init__()
        
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.output_frames = output_frames
        self.use_physics_projector = use_physics_projector
        
        # Main backbone: Typhoon-aware 3D UNet
        self.backbone = TyphoonAwareUNet3D(
            in_channels=latent_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            use_spiral_attention=use_spiral_attention,
            use_multiscale_temporal=use_multiscale_temporal
        )
        
        # Multi-task prediction heads
        self.structure_head = StructureHead(hidden_dim, latent_channels)
        self.track_head = TrackHead(hidden_dim, output_frames=output_frames)
        self.intensity_head = IntensityHead(hidden_dim, output_frames=output_frames)
        
        # Physics constraint module
        if use_physics_projector:
            self.physics_projector = PhysicsProjector(latent_channels)
        else:
            self.physics_projector = None
    
    def forward(self, z_noisy, t, condition_dict, return_x0=True):
        """
        Forward pass
        
        Args:
            z_noisy: (B, T, C_latent, H/8, W/8) noisy future latents
            t: (B,) diffusion timestep
            condition_dict: {
                'past_latents': (B, T_past, C_latent, H/8, W/8),
                'past_track': (B, T_past, 2),
                'past_intensity': (B, T_past)
            }
            return_x0: Whether to return predicted x0
        
        Returns:
            predictions: {
                'noise': predicted noise for diffusion,
                'track': (B, T, 2) predicted coordinates,
                'intensity': (B, T) predicted wind speeds,
                'x0_pred': (optional) predicted clean latents
            }
        """
        # Extract shared features using typhoon-aware backbone
        features = self.backbone(z_noisy, t, condition_dict)
        
        # Multi-task predictions
        noise_pred = self.structure_head(features)  # (B, T, C_latent, H/8, W/8)
        track_pred = self.track_head(features)       # (B, T, 2)
        intensity_pred = self.intensity_head(features)  # (B, T)
        
        predictions = {
            'noise': noise_pred,
            'track': track_pred,
            'intensity': intensity_pred
        }
        
        # Apply physics constraints if enabled and return_x0 is True
        if return_x0 and self.use_physics_projector:
            x0_pred = self.predict_x0(z_noisy, noise_pred, t)
            x0_constrained = self.physics_projector(x0_pred)
            
            # Recompute noise from constrained x0
            noise_constrained = self.recompute_noise(z_noisy, x0_constrained, t)
            
            predictions['noise'] = noise_constrained
            predictions['x0_pred'] = x0_constrained
        elif return_x0:
            x0_pred = self.predict_x0(z_noisy, noise_pred, t)
            predictions['x0_pred'] = x0_pred
        
        return predictions
    
    def predict_x0(self, z_t, noise_pred, t, alpha_bar=None, schedule=None):
        """
        Predict clean sample x0 from noisy sample and predicted noise
        
        Args:
            z_t: (B, T, C, H, W) noisy sample
            noise_pred: (B, T, C, H, W) predicted noise
            t: (B,) timestep
            alpha_bar: Optional precomputed alpha_bar values
            schedule: Optional schedule object (for EDM support)
        
        Returns:
            x0_pred: (B, T, C, H, W) predicted clean sample
        """
        # Check if using EDM schedule
        if schedule is not None and hasattr(schedule, 'get_sigma'):
            # EDM formulation: x_0 = x_t - sigma_t * noise
            sigma_t = schedule.get_sigma(t)  # (B,)
            while len(sigma_t.shape) < len(z_t.shape):
                sigma_t = sigma_t.unsqueeze(-1)
            x0_pred = z_t - sigma_t * noise_pred
            return x0_pred
        
        # Standard DDPM formulation
        if alpha_bar is None:
            # Compute alpha_bar (assume linear schedule)
            beta = torch.linspace(1e-4, 0.02, 1000, device=z_t.device, dtype=torch.float32)
            alpha = 1 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
        
        # Get alpha_bar for current timesteps
        alpha_bar_t = alpha_bar[t]  # (B,)
        
        # Reshape for broadcasting: (B, 1, 1, 1, 1)
        alpha_bar_t = alpha_bar_t[:, None, None, None, None]
        
        # Predict x0: x0 = (z_t - sqrt(1-alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        x0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        return x0_pred
    
    def recompute_noise(self, z_t, x0, t, alpha_bar=None):
        """
        Recompute noise from noisy sample and clean sample
        
        Args:
            z_t: (B, T, C, H, W) noisy sample
            x0: (B, T, C, H, W) clean sample
            t: (B,) timestep
            alpha_bar: Optional precomputed alpha_bar values
        
        Returns:
            noise: (B, T, C, H, W) noise
        """
        if alpha_bar is None:
            beta = torch.linspace(1e-4, 0.02, 1000, device=z_t.device, dtype=torch.float32)
            alpha = 1 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
        
        alpha_bar_t = alpha_bar[t][:, None, None, None, None]
        
        # Solve for noise: noise = (z_t - sqrt(alpha_bar_t) * x0) / sqrt(1-alpha_bar_t)
        noise = (z_t - torch.sqrt(alpha_bar_t) * x0) / torch.sqrt(1 - alpha_bar_t)
        
        return noise


class DiffusionSchedule:
    """
    Diffusion noise schedule (DDPM/DDIM)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'linear'
    ):
        """
        Initialize diffusion schedule
        
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: 'linear' or 'cosine'
        """
        self.num_timesteps = num_timesteps
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule_type == 'cosine':
            # Cosine schedule from "Improved DDPM"
            s = 0.008
            timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            alphas_bar = torch.cos(((timesteps / num_timesteps) + s) / (1 + s) * torch.tensor(torch.pi, dtype=torch.float32) * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            self.betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        
        # For DDIM sampling
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
    
    def add_noise(self, x0, t, noise=None):
        """
        Add noise to clean samples
        
        Args:
            x0: (B, ...) clean samples
            t: (B,) timesteps
            noise: Optional pre-generated noise
        
        Returns:
            noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_bar_t.shape) < len(x0.shape):
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)
        
        noisy = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        
        return noisy
    
    def to(self, device):
        """Move tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        self.sqrt_alphas_bar = self.sqrt_alphas_bar.to(device)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(device)
        return self


def test_model():
    """Test model forward pass"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PhysicsInformedDiffusionModel(
        latent_channels=8,
        hidden_dim=128,  # Smaller for testing
        num_heads=4,
        output_frames=12  # Predict 12 future timesteps
    ).to(device)
    
    # Test inputs
    B, T, C, H, W = 2, 12, 8, 32, 32  # T=12 for future frames
    T_past = 8  # 8 past timesteps for conditioning
    
    z_noisy = torch.randn(B, T, C, H, W).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    
    condition_dict = {
        'past_latents': torch.randn(B, T_past, C, H, W).to(device),
        'past_track': torch.randn(B, T_past, 2).to(device),
        'past_intensity': torch.randn(B, T_past).to(device)
    }
    
    # Forward pass
    predictions = model(z_noisy, t, condition_dict)
    
    print("Model output shapes:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Expected shapes
    assert predictions['noise'].shape == (B, T, C, H, W)
    assert predictions['track'].shape == (B, T, 2)
    assert predictions['intensity'].shape == (B, T)
    
    print("\nModel test passed!")
    
    # Test diffusion schedule
    schedule = DiffusionSchedule(num_timesteps=1000).to(device)
    
    x0 = torch.randn(B, T, C, H, W).to(device)
    t_test = torch.randint(0, 1000, (B,)).to(device)
    
    noisy = schedule.add_noise(x0, t_test)
    
    print(f"\nDiffusion schedule test:")
    print(f"  Clean shape: {x0.shape}")
    print(f"  Noisy shape: {noisy.shape}")
    print(f"  Timesteps: {t_test}")
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    test_model()

