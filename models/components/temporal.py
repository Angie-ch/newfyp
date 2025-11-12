"""
Temporal modeling components for multi-scale atmospheric processes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-scale temporal modeling to capture both fast and slow processes
    
    This is a key innovation for capturing different atmospheric time scales:
    - Fast: frame-to-frame dynamics (3-frame window)
    - Medium: 6-hour processes (5-frame window)
    - Slow: synoptic scale (9-frame window)
    """
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        
        self.channels = channels
        
        # Fast branch: frame-to-frame dynamics
        self.fast_conv = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            nn.Dropout(dropout)
        )
        
        # Medium branch: 6-hour processes
        self.medium_conv = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels,
                kernel_size=(5, 3, 3),
                padding=(2, 1, 1)
            ),
            nn.Dropout(dropout)
        )
        
        # Slow branch: synoptic scale
        self.slow_conv = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv3d(
                channels, channels,
                kernel_size=(9, 3, 3),
                padding=(4, 1, 1)
            ),
            nn.Dropout(dropout)
        )
        
        # Adaptive fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 3, channels, 1),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            (B, C, T, H, W) with multi-scale temporal information
        """
        # Process at different temporal scales
        fast = self.fast_conv(x)
        medium = self.medium_conv(x)
        slow = self.slow_conv(x)
        
        # Concatenate and fuse
        multi_scale = torch.cat([fast, medium, slow], dim=1)
        fused = self.fusion(multi_scale)
        
        return fused + x  # Residual connection


class TemporalAttention(nn.Module):
    """
    Temporal attention to model dependencies across time
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # QKV
        qkv = self.qkv(h)
        
        # Reshape: group spatial locations, attend over time
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, T, H * W)
        qkv = qkv.permute(1, 0, 2, 5, 4, 3)  # (3, B, heads, HW, T, head_dim)
        
        # Reshape to (3, B*heads*HW, T, head_dim)
        qkv = qkv.reshape(3, B * self.num_heads * H * W, T, self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Temporal attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bnd,bmd->bnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bnm,bmd->bnd', attn, v)
        
        # Reshape back
        out = out.reshape(B, self.num_heads, H * W, T, self.head_dim)
        out = out.permute(0, 1, 4, 3, 2).reshape(B, C, T, H, W)
        
        # Project
        out = self.proj(out)
        
        return out + x


class CausalTemporalConv(nn.Module):
    """
    Causal temporal convolution (doesn't look into future)
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Padding only on the left (past)
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        
        Returns:
            (B, C, T)
        """
        # Pad on the left
        x = F.pad(x, (self.padding, 0))
        
        return self.conv(x)


class TemporalEncoder(nn.Module):
    """
    Encode temporal sequence with causal convolutions
    """
    
    def __init__(
        self,
        channels: int,
        num_layers: int = 4,
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CausalTemporalConv(
                channels, 
                channels,
                kernel_size=kernel_size,
                dilation=2**i
            )
            for i in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(32, channels), channels)
            for _ in range(num_layers)
        ])
        
        self.act = nn.SiLU()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        
        Returns:
            (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # Reshape to process temporally
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        
        # Apply causal convolutions
        for layer, norm in zip(self.layers, self.norms):
            h = layer(x)
            h = h.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)
            h = norm(h)
            h = self.act(h)
            h = h.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
            
            x = x + h
        
        # Reshape back
        x = x.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)
        
        return x

