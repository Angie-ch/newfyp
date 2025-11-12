"""
Basic building blocks for neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    2D Residual block with GroupNorm and SiLU activation
    """
    
    def __init__(self, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class ResBlock3D(nn.Module):
    """
    3D Residual block for spatiotemporal processing
    """
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        
        self.act = nn.SiLU()
    
    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + x


class Downsample(nn.Module):
    """2D downsampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """2D upsampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample3D(nn.Module):
    """3D downsampling layer"""
    
    def __init__(self, channels: int, downsample_time: bool = False):
        super().__init__()
        
        if downsample_time:
            stride = (2, 2, 2)
        else:
            stride = (1, 2, 2)
        
        self.conv = nn.Conv3d(channels, channels, 3, stride=stride, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling layer"""
    
    def __init__(self, channels: int, upsample_time: bool = False):
        super().__init__()
        
        if upsample_time:
            self.scale_factor = (2, 2, 2)
        else:
            self.scale_factor = (1, 2, 2)
        
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return self.conv(x)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: (B,) timesteps
        
        Returns:
            (B, dim) embeddings
        """
        device = t.device
        half = self.dim // 2
        
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device)) * 
            torch.arange(0, half, dtype=torch.float32, device=device) / half
        )
        
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding


class AttentionBlock(nn.Module):
    """
    2D spatial attention block
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # QKV
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Project
        out = self.proj(out)
        
        return out + x

