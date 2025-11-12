"""
Spatial Autoencoder for Frame Compression

Compresses atmospheric fields spatially by factor of 8 (256×256 → 32×32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.blocks import ResBlock, Downsample, Upsample, AttentionBlock


class SpatialAutoencoder(nn.Module):
    """
    Compress spatial dimensions by factor of 8
    Input: (B, C, H, W) → Latent: (B, C_latent, H/8, W/8)
    
    Architecture:
    - Encoder: 4 stages with residual blocks and downsampling
    - Decoder: 4 stages with residual blocks and upsampling
    - Attention blocks at bottleneck for global context
    """
    
    def __init__(
        self,
        in_channels: int = 40,
        latent_channels: int = 8,
        hidden_dims: list = [64, 128, 256, 256],
        use_attention: bool = True
    ):
        """
        Initialize autoencoder
        
        Args:
            in_channels: Number of input channels (ERA5 variables)
            latent_channels: Number of latent channels
            hidden_dims: Hidden dimensions for each stage
            use_attention: Use attention at bottleneck
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        
        # ===== ENCODER =====
        
        # Initial convolution
        self.encoder_input = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Stage 1: 256x256 → 128x128
        self.encoder_stage1 = nn.ModuleList([
            ResBlock(hidden_dims[0], hidden_dims[0]),
            ResBlock(hidden_dims[0], hidden_dims[0]),
            Downsample(hidden_dims[0])
        ])
        
        # Stage 2: 128x128 → 64x64
        self.encoder_stage2 = nn.ModuleList([
            ResBlock(hidden_dims[0], hidden_dims[1]),
            ResBlock(hidden_dims[1], hidden_dims[1]),
            Downsample(hidden_dims[1])
        ])
        
        # Stage 3: 64x64 → 32x32
        self.encoder_stage3 = nn.ModuleList([
            ResBlock(hidden_dims[1], hidden_dims[2]),
            ResBlock(hidden_dims[2], hidden_dims[2]),
            Downsample(hidden_dims[2])
        ])
        
        # Bottleneck: 32x32
        self.encoder_bottleneck = nn.ModuleList([
            ResBlock(hidden_dims[2], hidden_dims[3]),
            AttentionBlock(hidden_dims[3]) if use_attention else nn.Identity(),
            ResBlock(hidden_dims[3], hidden_dims[3]),
        ])
        
        # To latent space
        self.to_latent = nn.Conv2d(hidden_dims[3], latent_channels, 1)
        
        # ===== DECODER =====
        
        # From latent space
        self.from_latent = nn.Conv2d(latent_channels, hidden_dims[3], 1)
        
        # Bottleneck: 32x32
        self.decoder_bottleneck = nn.ModuleList([
            ResBlock(hidden_dims[3], hidden_dims[3]),
            AttentionBlock(hidden_dims[3]) if use_attention else nn.Identity(),
            ResBlock(hidden_dims[3], hidden_dims[2]),
        ])
        
        # Stage 3: 32x32 → 64x64
        self.decoder_stage3 = nn.ModuleList([
            Upsample(hidden_dims[2]),
            ResBlock(hidden_dims[2], hidden_dims[2]),
            ResBlock(hidden_dims[2], hidden_dims[1]),
        ])
        
        # Stage 2: 64x64 → 128x128
        self.decoder_stage2 = nn.ModuleList([
            Upsample(hidden_dims[1]),
            ResBlock(hidden_dims[1], hidden_dims[1]),
            ResBlock(hidden_dims[1], hidden_dims[0]),
        ])
        
        # Stage 1: 128x128 → 256x256
        self.decoder_stage1 = nn.ModuleList([
            Upsample(hidden_dims[0]),
            ResBlock(hidden_dims[0], hidden_dims[0]),
            ResBlock(hidden_dims[0], hidden_dims[0]),
        ])
        
        # Output convolution
        self.decoder_output = nn.Conv2d(hidden_dims[0], in_channels, 3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights properly to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Kaiming initialization for Conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize output layer with smaller weights for stability
        nn.init.xavier_normal_(self.decoder_output.weight, gain=0.1)
        if self.decoder_output.bias is not None:
            nn.init.constant_(self.decoder_output.bias, 0)
    
    def encode(self, x):
        """
        Encode input to latent space
        
        Args:
            x: (B, C, H, W) input
        
        Returns:
            (B, C_latent, H/8, W/8) latent representation
        """
        # Initial conv
        h = self.encoder_input(x)
        
        # Stage 1
        for layer in self.encoder_stage1:
            h = layer(h)
        
        # Stage 2
        for layer in self.encoder_stage2:
            h = layer(h)
        
        # Stage 3
        for layer in self.encoder_stage3:
            h = layer(h)
        
        # Bottleneck
        for layer in self.encoder_bottleneck:
            h = layer(h)
        
        # To latent
        z = self.to_latent(h)
        
        return z
    
    def decode(self, z):
        """
        Decode latent to reconstruction
        
        Args:
            z: (B, C_latent, H/8, W/8) latent
        
        Returns:
            (B, C, H, W) reconstruction
        """
        # From latent
        h = self.from_latent(z)
        
        # Bottleneck
        for layer in self.decoder_bottleneck:
            h = layer(h)
        
        # Stage 3
        for layer in self.decoder_stage3:
            h = layer(h)
        
        # Stage 2
        for layer in self.decoder_stage2:
            h = layer(h)
        
        # Stage 1
        for layer in self.decoder_stage1:
            h = layer(h)
        
        # Output
        recon = self.decoder_output(h)
        
        return recon
    
    def forward(self, x):
        """
        Full forward pass
        
        Args:
            x: (B, C, H, W) input
        
        Returns:
            recon: (B, C, H, W) reconstruction
            z: (B, C_latent, H/8, W/8) latent
        """
        z = self.encode(x)
        recon = self.decode(z)
        
        return recon, z
    
    def get_latent_shape(self, input_shape):
        """
        Calculate latent shape from input shape
        
        Args:
            input_shape: (H, W) input spatial dimensions
        
        Returns:
            (H/8, W/8) latent spatial dimensions
        """
        H, W = input_shape
        return (H // 8, W // 8)


class AutoencoderLoss(nn.Module):
    """
    Loss function for autoencoder training
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.0
    ):
        """
        Initialize loss
        
        Args:
            recon_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss (if > 0)
        """
        super().__init__()
        
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
    
    def forward(self, recon, target):
        """
        Compute loss
        
        Args:
            recon: (B, C, H, W) reconstruction
            target: (B, C, H, W) target
        
        Returns:
            loss: Scalar loss
            loss_dict: Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        loss_recon = F.mse_loss(recon, target)
        
        # Total loss
        total_loss = self.recon_weight * loss_recon
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': loss_recon.item(),
        }
        
        # Perceptual loss (optional, could add VGG-based loss)
        if self.perceptual_weight > 0:
            # For atmospheric data, perceptual loss is less standard
            # Could be replaced with physics-based metrics
            pass
        
        return total_loss, loss_dict


def test_autoencoder():
    """Test autoencoder forward pass"""
    
    # Create model
    model = SpatialAutoencoder(
        in_channels=40,
        latent_channels=8,
        hidden_dims=[64, 128, 256, 256]
    )
    
    # Test input
    x = torch.randn(2, 40, 256, 256)
    
    # Forward pass
    recon, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    # Test encode/decode separately
    z2 = model.encode(x)
    recon2 = model.decode(z2)
    
    assert torch.allclose(z, z2)
    assert torch.allclose(recon, recon2)
    
    print("Autoencoder test passed!")


if __name__ == '__main__':
    test_autoencoder()

