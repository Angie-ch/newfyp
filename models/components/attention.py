"""
Attention mechanisms for typhoon prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Standard self-attention mechanism
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, ...) any shape
        
        Returns:
            Same shape as input
        """
        original_shape = x.shape
        B, C = x.shape[:2]
        
        # Reshape to (B, C, N)
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        
        # Normalize
        h = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # QKV
        qkv = self.qkv(h)  # (B, N, 3*C)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, -1, C)
        
        # Project
        out = self.proj(out)
        
        # Reshape back
        out = out.permute(0, 2, 1).reshape(original_shape)
        
        return out + x.permute(0, 2, 1).reshape(original_shape)


class CrossAttention3D(nn.Module):
    """
    Cross-attention for conditioning on past frames
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        # Use fewer groups for better compatibility with small spatial dimensions
        num_groups = min(8, channels // 4) if channels >= 8 else 1
        self.norm_q = nn.GroupNorm(num_groups, channels)
        self.norm_kv = nn.GroupNorm(num_groups, channels)
        
        self.q = nn.Conv3d(channels, channels, 1)
        self.kv = nn.Conv3d(channels, channels * 2, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, C, T, H, W) query
            context: (B, C, T_context, H, W) key/value, if None use x
        
        Returns:
            (B, C, T, H, W)
        """
        if context is None:
            context = x
        
        B, C, T, H, W = x.shape
        _, _, T_ctx, _, _ = context.shape
        
        # Normalize
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        
        # Project
        q = self.q(q)  # (B, C, T, H, W)
        kv = self.kv(kv)  # (B, 2*C, T_ctx, H, W)
        
        # Reshape for attention
        q = q.reshape(B, self.num_heads, self.head_dim, T * H * W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, THW, head_dim)
        
        kv = kv.reshape(B, 2, self.num_heads, self.head_dim, T_ctx * H * W)
        kv = kv.permute(1, 0, 2, 4, 3)  # (2, B, heads, T_ctx*HW, head_dim)
        k, v = kv[0], kv[1]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, T, H, W)
        
        # Project
        out = self.proj(out)
        
        return out + x


class SpiralAttentionBlock(nn.Module):
    """
    Spiral attention mechanism biased toward cyclone patterns
    
    This is a key innovation for typhoon-aware modeling
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
        
        # Learnable spiral bias parameters
        self.spiral_strength = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * 0.1)
        self.spiral_frequency = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * 10.0)
    
    def forward(self, x, center_coords=None):
        """
        Args:
            x: (B, C, T, H, W) feature map
            center_coords: (B, T, 2) typhoon center coordinates (optional)
        
        Returns:
            (B, C, T, H, W) with spiral-aware attention
        """
        B, C, T, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # QKV
        qkv = self.qkv(h)  # (B, 3*C, T, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, T, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 6, 3)  # (3, B, heads, T, H, W, head_dim)
        
        # Reshape for attention: (B, heads, T*H*W, head_dim)
        qkv = qkv.reshape(3, B, self.num_heads, T * H * W, self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        
        # Add spiral bias
        if center_coords is not None:
            spiral_bias = self._create_spiral_bias(
                B, T, H, W, 
                center_coords, 
                x.device
            )
            attn = attn + spiral_bias.unsqueeze(1)  # Add head dimension
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.reshape(B, self.num_heads, T, H, W, self.head_dim)
        out = out.permute(0, 1, 5, 2, 3, 4).reshape(B, C, T, H, W)
        
        # Project
        out = self.proj(out)
        
        return out + x
    
    def _create_spiral_bias(
        self, 
        B: int, 
        T: int, 
        H: int, 
        W: int,
        center_coords: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create attention bias favoring spiral patterns
        
        Args:
            B: Batch size
            T: Temporal dimension
            H, W: Spatial dimensions
            center_coords: (B, T, 2) center positions in [0, 1] range
            device: torch device
        
        Returns:
            (B, T*H*W, T*H*W) attention bias
        """
        # Create spatial grid
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Expand for batch and time
        grid_y = grid_y[None, None, :, :].expand(B, T, -1, -1)
        grid_x = grid_x[None, None, :, :].expand(B, T, -1, -1)
        
        # Compute relative positions to center
        if center_coords.shape[-1] == 2:
            # Normalize center coords if needed (assume they're in pixel coordinates)
            center_y = center_coords[:, :, 0:1, None, None]  # (B, T, 1, 1, 1)
            center_x = center_coords[:, :, 1:2, None, None]  # (B, T, 1, 1, 1)
        else:
            center_y = 0.5
            center_x = 0.5
        
        # Compute polar coordinates relative to center
        dy = grid_y - center_y.squeeze(-1).squeeze(-1)
        dx = grid_x - center_x.squeeze(-1).squeeze(-1)
        
        r = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)
        
        # Spiral pattern: points along spiral arms get higher bias
        # Using parameterized spiral equation
        spiral_score = torch.zeros(B, T, H, W, device=device, dtype=torch.float32)
        
        for h in range(self.num_heads):
            freq = self.spiral_frequency[h]
            strength = self.spiral_strength[h]
            spiral_score += strength * torch.cos(theta - freq * r)
        
        spiral_score = spiral_score / self.num_heads
        
        # Create pairwise bias
        # For simplicity, use same-timestep spatial bias
        spiral_score = spiral_score.reshape(B, T * H * W)
        
        # Create distance-based bias (closer points attend more)
        bias = spiral_score.unsqueeze(2) + spiral_score.unsqueeze(1)
        bias = bias * 0.1  # Scale factor
        
        return bias


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary position embedding for improved position encoding
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache
        self.max_seq_len = max_seq_len
        self._cached_freqs = None
        self._cached_len = 0
    
    def forward(self, seq_len: int, device: torch.device):
        """
        Args:
            seq_len: Sequence length
            device: torch device
        
        Returns:
            (seq_len, dim) rotation matrix
        """
        if seq_len > self._cached_len or self._cached_freqs is None:
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cached_freqs = emb
            self._cached_len = seq_len
        
        return self._cached_freqs[:seq_len]

