"""
Joint Autoencoder for ERA5 + IBTrACS

Architecture:
1. ENCODER: Fuses ERA5 atmospheric fields + IBTrACS track/intensity → Unified Latent
2. DECODER: Splits Unified Latent → ERA5 (spatial) + IBTrACS (scalars) separately

This allows:
- Joint representation learning in latent space
- Diffusion operates on unified latent containing all information
- Separate reconstruction for each modality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.blocks import ResBlock, Downsample, Upsample, AttentionBlock


class JointAutoencoder(nn.Module):
    """
    Joint autoencoder that encodes ERA5 + IBTrACS together into unified latent space,
    then decodes them separately for reconstruction and prediction.
    
    Flow:
        ERA5 (40, 64, 64) + IBTrACS (3,) → Encode → Latent (8, 8, 8)
        Latent (8, 8, 8) → Decode → ERA5 (40, 64, 64) + IBTrACS (3,)
    """
    
    def __init__(
        self,
        era5_channels: int = 40,
        latent_channels: int = 8,
        hidden_dims: list = None,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]
        
        self.era5_channels = era5_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # ═════════════════════════════════════════════════════════════
        # ENCODER: Fuse ERA5 + IBTrACS into unified latent
        # ═════════════════════════════════════════════════════════════
        
        # Step 1: Embed IBTrACS scalars (lat, lon, intensity) into feature space
        self.ibtracs_embedder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256)
        )
        
        # Step 2: Project IBTrACS features to spatial feature maps (16 channels, 64x64)
        self.ibtracs_to_spatial = nn.Sequential(
            nn.Linear(256, 64 * 64 * 16),
            nn.Unflatten(1, (16, 64, 64))
        )
        
        # Step 3: Joint encoder - processes concatenated ERA5 + IBTrACS_spatial
        # Input: ERA5 (40) + IBTrACS_spatial (16) = 56 channels
        joint_in_channels = era5_channels + 16
        
        self.encoder_input = nn.Conv2d(joint_in_channels, hidden_dims[0], 3, padding=1)
        
        # Encoder Stage 1: 64x64 → 32x32
        self.encoder_stage1 = nn.ModuleList([
            ResBlock(hidden_dims[0], hidden_dims[0], dropout=dropout),
            ResBlock(hidden_dims[0], hidden_dims[0], dropout=dropout),
            Downsample(hidden_dims[0])
        ])
        
        # Encoder Stage 2: 32x32 → 16x16
        self.encoder_stage2 = nn.ModuleList([
            ResBlock(hidden_dims[0], hidden_dims[1], dropout=dropout),
            ResBlock(hidden_dims[1], hidden_dims[1], dropout=dropout),
            Downsample(hidden_dims[1])
        ])
        
        # Encoder Stage 3: 16x16 → 8x8
        self.encoder_stage3 = nn.ModuleList([
            ResBlock(hidden_dims[1], hidden_dims[2], dropout=dropout),
            ResBlock(hidden_dims[2], hidden_dims[2], dropout=dropout),
            Downsample(hidden_dims[2])
        ])
        
        # Bottleneck at 8x8
        self.encoder_bottleneck = nn.ModuleList([
            ResBlock(hidden_dims[2], hidden_dims[3], dropout=dropout),
            AttentionBlock(hidden_dims[3]) if use_attention else nn.Identity(),
            ResBlock(hidden_dims[3], hidden_dims[3], dropout=dropout),
        ])
        
        # Project to unified latent space
        self.to_latent = nn.Conv2d(hidden_dims[3], latent_channels, 1)
        
        # ═════════════════════════════════════════════════════════════
        # DECODER: Split unified latent → ERA5 + IBTrACS separately
        # ═════════════════════════════════════════════════════════════
        
        # From unified latent space
        self.from_latent = nn.Conv2d(latent_channels, hidden_dims[3], 1)
        
        # Shared decoder backbone
        self.decoder_bottleneck = nn.ModuleList([
            ResBlock(hidden_dims[3], hidden_dims[3], dropout=dropout),
            AttentionBlock(hidden_dims[3]) if use_attention else nn.Identity(),
            ResBlock(hidden_dims[3], hidden_dims[2], dropout=dropout),
        ])
        
        # Decoder Stage 3: 8x8 → 16x16
        self.decoder_stage3 = nn.ModuleList([
            Upsample(hidden_dims[2]),
            ResBlock(hidden_dims[2], hidden_dims[2], dropout=dropout),
            ResBlock(hidden_dims[2], hidden_dims[1], dropout=dropout),
        ])
        
        # Decoder Stage 2: 16x16 → 32x32
        self.decoder_stage2 = nn.ModuleList([
            Upsample(hidden_dims[1]),
            ResBlock(hidden_dims[1], hidden_dims[1], dropout=dropout),
            ResBlock(hidden_dims[1], hidden_dims[0], dropout=dropout),
        ])
        
        # Decoder Stage 1: 32x32 → 64x64
        self.decoder_stage1 = nn.ModuleList([
            Upsample(hidden_dims[0]),
            ResBlock(hidden_dims[0], hidden_dims[0], dropout=dropout),
            ResBlock(hidden_dims[0], hidden_dims[0], dropout=dropout),
        ])
        
        # ═════════════════════════════════════════════════════════════
        # SEPARATE OUTPUT HEADS
        # ═════════════════════════════════════════════════════════════
        
        # Head 1: ERA5 reconstruction (spatial output)
        self.era5_head = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], era5_channels, 3, padding=1)
        )
        
        # Head 2: IBTrACS reconstruction (scalar output)
        self.ibtracs_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pool spatial: (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(hidden_dims[0], 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # Output: lat, lon, intensity
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Smaller initialization for output layers
        for m in self.era5_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
        
        for m in self.ibtracs_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
    
    def encode(self, era5, track, intensity):
        """
        Encode ERA5 + IBTrACS into unified latent space
        
        Args:
            era5: (B, C_era5, H, W) - atmospheric fields, e.g., (B, 40, 64, 64)
            track: (B, 2) - (lat, lon)
            intensity: (B,) or (B, 1) - wind speed
        
        Returns:
            z: (B, C_latent, H/8, W/8) - unified latent, e.g., (B, 8, 8, 8)
        """
        B = era5.shape[0]
        
        # Ensure intensity has shape (B, 1)
        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(1)
        
        # Step 1: Concatenate IBTrACS scalars
        ibtracs_scalars = torch.cat([track, intensity], dim=1)  # (B, 3)
        
        # Step 2: Embed IBTrACS into feature space
        ibtracs_feat = self.ibtracs_embedder(ibtracs_scalars)  # (B, 256)
        
        # Step 3: Project to spatial feature maps
        ibtracs_spatial = self.ibtracs_to_spatial(ibtracs_feat)  # (B, 16, 64, 64)
        
        # Step 4: Concatenate with ERA5 along channel dimension
        # Expected: era5 (B, 64, 64, 64) + ibtracs_spatial (B, 16, 64, 64) = (B, 80, 64, 64)
        if era5.shape[1] != self.era5_channels:
            raise ValueError(f"ERA5 input has {era5.shape[1]} channels, but model expects {self.era5_channels}")
        if ibtracs_spatial.shape[1] != 16:
            raise ValueError(f"IBTrACS spatial has {ibtracs_spatial.shape[1]} channels, expected 16")
        if era5.shape[2:] != ibtracs_spatial.shape[2:]:
            raise ValueError(f"Shape mismatch: era5 {era5.shape[2:]}, ibtracs_spatial {ibtracs_spatial.shape[2:]}")
        
        joint_input = torch.cat([era5, ibtracs_spatial], dim=1)  # (B, 80, 64, 64)
        
        # Step 5: Encode through CNN
        h = self.encoder_input(joint_input)
        
        # Encoder stages
        for layer in self.encoder_stage1:
            h = layer(h)
        
        for layer in self.encoder_stage2:
            h = layer(h)
        
        for layer in self.encoder_stage3:
            h = layer(h)
        
        # Bottleneck
        for layer in self.encoder_bottleneck:
            h = layer(h)
        
        # To latent space
        z = self.to_latent(h)  # (B, latent_channels, 8, 8)
        
        return z
    
    def decode(self, z):
        """
        Decode unified latent into SEPARATE ERA5 and IBTrACS
        
        Args:
            z: (B, C_latent, H/8, W/8) - unified latent, e.g., (B, 8, 8, 8)
        
        Returns:
            era5_recon: (B, C_era5, H, W) - reconstructed atmospheric fields
            track_recon: (B, 2) - reconstructed (lat, lon)
            intensity_recon: (B,) - reconstructed wind speed
        """
        # From latent space
        h = self.from_latent(z)
        
        # Shared decoder backbone
        for layer in self.decoder_bottleneck:
            h = layer(h)
        
        for layer in self.decoder_stage3:
            h = layer(h)
        
        for layer in self.decoder_stage2:
            h = layer(h)
        
        for layer in self.decoder_stage1:
            h = layer(h)
        
        # Now h is (B, hidden_dims[0], 64, 64) - shared features
        
        # ═════════════════════════════════════════════════════════════
        # SPLIT INTO SEPARATE OUTPUTS
        # ═════════════════════════════════════════════════════════════
        
        # Output 1: ERA5 (spatial)
        era5_recon = self.era5_head(h)  # (B, 40, 64, 64)
        
        # Output 2: IBTrACS (scalars)
        ibtracs_recon = self.ibtracs_head(h)  # (B, 3)
        
        # Split IBTrACS into track and intensity
        track_recon = ibtracs_recon[:, :2]  # (B, 2) - lat, lon
        intensity_recon = ibtracs_recon[:, 2]  # (B,) - wind speed
        
        return era5_recon, track_recon, intensity_recon
    
    def forward(self, era5, track, intensity):
        """
        Full forward pass: encode then decode
        
        Args:
            era5: (B, C_era5, H, W)
            track: (B, 2)
            intensity: (B,) or (B, 1)
        
        Returns:
            Dictionary with all outputs
        """
        # Encode to unified latent
        z = self.encode(era5, track, intensity)
        
        # Decode to separate outputs
        era5_recon, track_recon, intensity_recon = self.decode(z)
        
        return {
            'latent': z,
            'era5_recon': era5_recon,
            'track_recon': track_recon,
            'intensity_recon': intensity_recon
        }


class JointAutoencoderLoss(nn.Module):
    """
    Multi-task loss for joint autoencoder
    
    Balances reconstruction quality across:
    - ERA5 spatial fields (MSE)
    - Track coordinates (MSE with higher weight)
    - Intensity values (MSE with higher weight)
    """
    
    def __init__(
        self,
        era5_weight: float = 1.0,
        track_weight: float = 10.0,      # Higher weight for track
        intensity_weight: float = 5.0,    # Higher weight for intensity
    ):
        super().__init__()
        self.era5_weight = era5_weight
        self.track_weight = track_weight
        self.intensity_weight = intensity_weight
    
    def forward(self, outputs, targets):
        """
        Compute multi-task reconstruction loss
        
        Args:
            outputs: dict with 'era5_recon', 'track_recon', 'intensity_recon'
            targets: dict with 'era5', 'track', 'intensity'
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        # ERA5 reconstruction loss (spatial)
        loss_era5 = F.mse_loss(outputs['era5_recon'], targets['era5'])
        
        # Track reconstruction loss (lat, lon)
        loss_track = F.mse_loss(outputs['track_recon'], targets['track'])
        
        # Intensity reconstruction loss (wind speed)
        loss_intensity = F.mse_loss(outputs['intensity_recon'], targets['intensity'])
        
        # Weighted total loss
        total_loss = (
            self.era5_weight * loss_era5 +
            self.track_weight * loss_track +
            self.intensity_weight * loss_intensity
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'era5': loss_era5.item(),
            'track': loss_track.item(),
            'intensity': loss_intensity.item()
        }
        
        return total_loss, loss_dict

