"""
Multi-Task Prediction Heads

Predict structure, track, and intensity simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureHead(nn.Module):
    """
    Predict spatial structure (noise for diffusion)
    """
    
    def __init__(self, hidden_dim: int, out_channels: int):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv3d(hidden_dim, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, out_channels, 3, padding=1)
        )
    
    def forward(self, features):
        """
        Args:
            features: (B, hidden_dim, T, H, W)
        
        Returns:
            noise_pred: (B, T, C, H, W) predicted noise
        """
        noise_pred = self.head(features)
        
        # Rearrange to (B, T, C, H, W)
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
        
        return noise_pred


class TrackHead(nn.Module):
    """
    Predict typhoon center trajectory
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_frames: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_frames = output_frames
        
        # Global pooling to aggregate spatial information
        self.global_pool = nn.AdaptiveAvgPool3d((output_frames, 1, 1))
        
        # MLP for coordinate prediction
        layers = []
        in_dim = hidden_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim // 2
        
        layers.append(nn.Linear(in_dim, 2))  # (lat, lon)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Args:
            features: (B, hidden_dim, T, H, W)
        
        Returns:
            track: (B, T, 2) predicted (lat, lon) for each frame
        """
        B, C, T, H, W = features.shape
        
        # Pool spatially, keep temporal
        pooled = self.global_pool(features)  # (B, C, T, 1, 1)
        pooled = pooled.squeeze(-1).squeeze(-1)  # (B, C, T)
        pooled = pooled.permute(0, 2, 1)  # (B, T, C)
        
        # Predict coordinates for each timestep
        track = []
        for t in range(T):
            coords = self.mlp(pooled[:, t, :])  # (B, 2)
            track.append(coords)
        
        track = torch.stack(track, dim=1)  # (B, T, 2)
        
        return track


class IntensityHead(nn.Module):
    """
    Predict maximum wind speed (intensity)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_frames: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_frames = output_frames
        
        # Global max pooling (intensity = max wind)
        self.global_pool = nn.AdaptiveMaxPool3d((output_frames, 1, 1))
        
        # MLP for intensity prediction
        layers = []
        in_dim = hidden_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim // 2
        
        layers.append(nn.Linear(in_dim, 1))  # Single value
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Args:
            features: (B, hidden_dim, T, H, W)
        
        Returns:
            intensity: (B, T) predicted wind speeds
        """
        B, C, T, H, W = features.shape
        
        # Pool spatially with max (captures peak intensity)
        pooled = self.global_pool(features)  # (B, C, T, 1, 1)
        pooled = pooled.squeeze(-1).squeeze(-1)  # (B, C, T)
        pooled = pooled.permute(0, 2, 1)  # (B, T, C)
        
        # Predict intensity for each timestep
        intensity = []
        for t in range(T):
            wind = self.mlp(pooled[:, t, :])  # (B, 1)
            intensity.append(wind)
        
        intensity = torch.stack(intensity, dim=1).squeeze(-1)  # (B, T)
        
        return intensity


class PressureHead(nn.Module):
    """
    Predict minimum central pressure (optional auxiliary task)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_frames: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_frames = output_frames
        
        # Global min pooling (pressure = minimum value)
        self.global_pool = nn.AdaptiveMaxPool3d((output_frames, 1, 1))
        
        # MLP for pressure prediction
        layers = []
        in_dim = hidden_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim // 2
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Args:
            features: (B, hidden_dim, T, H, W)
        
        Returns:
            pressure: (B, T) predicted pressures
        """
        B, C, T, H, W = features.shape
        
        # Pool spatially
        pooled = self.global_pool(features)
        pooled = pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        # Predict pressure
        pressure = []
        for t in range(T):
            p = self.mlp(pooled[:, t, :])
            pressure.append(p)
        
        pressure = torch.stack(pressure, dim=1).squeeze(-1)
        
        return pressure

