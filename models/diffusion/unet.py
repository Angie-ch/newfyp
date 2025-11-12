"""
Typhoon-Aware 3D UNet Backbone

Incorporates spiral attention and multi-scale temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..components.blocks import ResBlock3D, Downsample3D, Upsample3D, TimeEmbedding
from ..components.attention import SpiralAttentionBlock, CrossAttention3D
from ..components.temporal import MultiScaleTemporalBlock


class TyphoonAwareUNet3D(nn.Module):
    """
    3D UNet with typhoon-specific components:
    - Spiral attention for cyclone patterns
    - Multi-scale temporal modeling
    - Cross-attention to past frames for conditioning
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        depth: int = 3,
        use_spiral_attention: bool = True,
        use_multiscale_temporal: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # Time embedding for diffusion timestep
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Conditioning encoder (processes past frames)
        self.cond_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        # Project conditioning features to match bottleneck dimension for cross-attention
        self.cond_proj = nn.Conv3d(64, hidden_dim, 1)
        
        # ===== ENCODER =====
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, 64, 3, padding=1)
        
        # Encoder level 1: Full resolution
        enc1_layers = [
            ResBlock3D(64),
            ResBlock3D(64)
        ]
        if use_multiscale_temporal:
            enc1_layers.append(MultiScaleTemporalBlock(64))
        if use_spiral_attention:
            enc1_layers.append(SpiralAttentionBlock(64, num_heads=num_heads))
        
        self.enc1 = nn.Sequential(*enc1_layers)
        
        # Encoder level 2: 1/2 resolution
        self.down1 = Downsample3D(64, downsample_time=False)
        
        enc2_layers = [
            ResBlock3D(64),
            nn.Conv3d(64, 128, 1),
            ResBlock3D(128)
        ]
        if use_multiscale_temporal:
            enc2_layers.append(MultiScaleTemporalBlock(128))
        if use_spiral_attention:
            enc2_layers.append(SpiralAttentionBlock(128, num_heads=num_heads))
        
        self.enc2 = nn.Sequential(*enc2_layers)
        
        # Encoder level 3: 1/4 resolution
        self.down2 = Downsample3D(128, downsample_time=False)
        
        enc3_layers = [
            ResBlock3D(128),
            nn.Conv3d(128, 256, 1),
            ResBlock3D(256)
        ]
        if use_multiscale_temporal:
            enc3_layers.append(MultiScaleTemporalBlock(256))
        
        self.enc3 = nn.Sequential(*enc3_layers)
        
        # ===== BOTTLENECK =====
        
        self.down3 = Downsample3D(256, downsample_time=False)
        
        self.bottleneck = nn.Sequential(
            ResBlock3D(256),
            nn.Conv3d(256, hidden_dim, 1),
            ResBlock3D(hidden_dim),
            CrossAttention3D(hidden_dim, num_heads=num_heads),  # Attend to conditioning
            ResBlock3D(hidden_dim)
        )
        
        # ===== DECODER =====
        
        # Decoder level 3
        self.up3 = Upsample3D(hidden_dim, upsample_time=False)
        
        dec3_layers = [
            nn.Conv3d(hidden_dim + 256, 256, 1),  # +skip connection
            ResBlock3D(256),
            ResBlock3D(256)
        ]
        if use_multiscale_temporal:
            dec3_layers.append(MultiScaleTemporalBlock(256))
        
        self.dec3 = nn.Sequential(*dec3_layers)
        
        # Decoder level 2
        self.up2 = Upsample3D(256, upsample_time=False)
        
        dec2_layers = [
            nn.Conv3d(256 + 128, 128, 1),  # +skip connection
            ResBlock3D(128),
            ResBlock3D(128)
        ]
        if use_multiscale_temporal:
            dec2_layers.append(MultiScaleTemporalBlock(128))
        if use_spiral_attention:
            dec2_layers.append(SpiralAttentionBlock(128, num_heads=num_heads))
        
        self.dec2 = nn.Sequential(*dec2_layers)
        
        # Decoder level 1
        self.up1 = Upsample3D(128, upsample_time=False)
        
        dec1_layers = [
            nn.Conv3d(128 + 64, 128, 1),  # +skip connection
            ResBlock3D(128),
            ResBlock3D(128)
        ]
        if use_multiscale_temporal:
            dec1_layers.append(MultiScaleTemporalBlock(128))
        if use_spiral_attention:
            dec1_layers.append(SpiralAttentionBlock(128, num_heads=num_heads))
        
        self.dec1 = nn.Sequential(*dec1_layers)
        
        # Output projection
        self.output_proj = nn.Conv3d(128, hidden_dim, 3, padding=1)
    
    def forward(self, z_noisy, t, condition_dict):
        """
        Args:
            z_noisy: (B, T, C_latent, H/8, W/8) noisy future latents
            t: (B,) diffusion timestep
            condition_dict: {
                'past_latents': (B, T_past, C_latent, H/8, W/8),
                'past_track': (B, T_past, 2),
                'past_intensity': (B, T_past)
            }
        
        Returns:
            features: (B, hidden_dim, T, H/8, W/8) for prediction heads
        """
        # Rearrange to (B, C, T, H, W)
        x = z_noisy.permute(0, 2, 1, 3, 4)
        
        # Time embedding
        t_emb = self.get_timestep_embedding(t, 128, x.device)
        t_emb = self.time_embed(t_emb)  # (B, hidden_dim)
        
        # Condition encoding
        cond = condition_dict['past_latents'].permute(0, 2, 1, 3, 4)
        cond_feat = self.cond_encoder(cond)  # (B, 64, T_past, H, W)
        cond_feat = self.cond_proj(cond_feat)  # (B, hidden_dim, T_past, H, W)
        
        # ===== ENCODER =====
        
        x = self.init_conv(x)
        
        # Level 1
        e1 = self.enc1(x)  # (B, 64, T, H, W)
        
        # Level 2
        x = self.down1(e1)
        e2 = self.enc2(x)  # (B, 128, T, H/2, W/2)
        
        # Level 3
        x = self.down2(e2)
        e3 = self.enc3(x)  # (B, 256, T, H/4, W/4)
        
        # ===== BOTTLENECK =====
        
        x = self.down3(e3)
        
        # Process bottleneck with conditioning
        for i, layer in enumerate(self.bottleneck):
            if isinstance(layer, CrossAttention3D):
                # Cross-attend to past frames
                # Resize cond_feat to match spatial dimensions of x
                B, C_cond, T_past, H_cond, W_cond = cond_feat.shape
                _, _, T_x, H_x, W_x = x.shape
                
                # Reshape to (B*T_past, C_cond, H_cond, W_cond) for 2D interpolation
                cond_flat = cond_feat.permute(0, 2, 1, 3, 4).reshape(B * T_past, C_cond, H_cond, W_cond)
                
                # Resize spatially to match x's spatial dimensions
                cond_resized = F.interpolate(cond_flat, size=(H_x, W_x), mode='nearest')
                
                # Reshape back to (B, C_cond, T_past, H_x, W_x)
                cond_resized = cond_resized.reshape(B, T_past, C_cond, H_x, W_x).permute(0, 2, 1, 3, 4)
                
                x = layer(x, cond_resized)
            else:
                x = layer(x)
        
        b = x  # (B, hidden_dim, T, H/8, W/8)
        
        # ===== DECODER =====
        
        # Level 3
        x = self.up3(b)
        x = torch.cat([x, e3], dim=1)  # Skip connection
        x = self.dec3(x)
        
        # Level 2
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)  # Skip connection
        x = self.dec2(x)
        
        # Level 1
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)  # Skip connection
        x = self.dec1(x)
        
        # Output
        features = self.output_proj(x)
        
        # Add time embedding
        features = features + t_emb[:, :, None, None, None]
        
        return features  # (B, hidden_dim, T, H, W)
    
    def get_timestep_embedding(self, timesteps, dim, device):
        """
        Sinusoidal timestep embedding
        
        Args:
            timesteps: (B,) timesteps
            dim: Embedding dimension
            device: torch device
        
        Returns:
            (B, dim) embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding


def test_unet():
    """Test UNet forward pass"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TyphoonAwareUNet3D(
        in_channels=8,
        hidden_dim=256,
        num_heads=8
    ).to(device)
    
    # Test inputs
    B, T, C, H, W = 2, 8, 8, 32, 32
    T_past = 12
    
    z_noisy = torch.randn(B, T, C, H, W).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    
    condition_dict = {
        'past_latents': torch.randn(B, T_past, C, H, W).to(device),
        'past_track': torch.randn(B, T_past, 2).to(device),
        'past_intensity': torch.randn(B, T_past).to(device)
    }
    
    # Forward pass
    features = model(z_noisy, t, condition_dict)
    
    print(f"Input shape: {z_noisy.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Expected: (B={B}, hidden_dim=256, T={T}, H={H}, W={W})")
    
    assert features.shape == (B, 256, T, H, W)
    
    print("UNet test passed!")


if __name__ == '__main__':
    test_unet()

