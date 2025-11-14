"""
EDM (Elucidated Diffusion Models) Noise Schedule

Based on: "Elucidating the Design Space of Diffusion-Based Generative Models"
https://arxiv.org/abs/2206.00364

Key improvements:
- Better parameterization using sigma (noise level) instead of beta
- Improved training dynamics
- Better sampling quality
"""

import torch
import torch.nn.functional as F
import numpy as np


class EDMSchedule:
    """
    EDM-style noise schedule for diffusion models
    
    Uses sigma (noise level) parameterization instead of beta schedule.
    This provides better training dynamics and sampling quality.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_timesteps: int = 1000
    ):
        """
        Initialize EDM schedule
        
        Args:
            sigma_min: Minimum noise level (close to clean data)
            sigma_max: Maximum noise level (pure noise)
            rho: Controls the schedule curve (higher = more emphasis on high noise)
            num_timesteps: Number of diffusion timesteps
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_timesteps = num_timesteps
        
        # Precompute sigma values for each timestep
        # EDM schedule: sigma(t) = (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        t = torch.linspace(0, 1, num_timesteps, dtype=torch.float32)
        sigma_max_inv_rho = sigma_max ** (1.0 / rho)
        sigma_min_inv_rho = sigma_min ** (1.0 / rho)
        
        self.sigmas = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** rho
        
        # For compatibility with DDPM-style code, also compute alphas
        # In EDM, we use: x_t = x_0 + sigma_t * noise
        # In DDPM, we use: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # So: sqrt(alpha_bar_t) = 1 / sqrt(1 + sigma_t^2)
        #     sqrt(1 - alpha_bar_t) = sigma_t / sqrt(1 + sigma_t^2)
        self.sqrt_alphas_bar = 1.0 / torch.sqrt(1.0 + self.sigmas ** 2)
        self.sqrt_one_minus_alphas_bar = self.sigmas / torch.sqrt(1.0 + self.sigmas ** 2)
        
        # For denoising
        self.alphas_bar = self.sqrt_alphas_bar ** 2
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get sigma (noise level) for given timesteps
        
        Args:
            t: (B,) timestep indices [0, num_timesteps-1]
        
        Returns:
            sigma: (B,) noise levels
        """
        return self.sigmas[t]
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Add noise to clean samples using EDM parameterization
        
        Args:
            x0: (B, ...) clean samples
            t: (B,) timestep indices
            noise: Optional pre-generated noise
        
        Returns:
            noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Get sigma for each sample
        sigma_t = self.get_sigma(t)  # (B,)
        
        # Reshape for broadcasting
        while len(sigma_t.shape) < len(x0.shape):
            sigma_t = sigma_t.unsqueeze(-1)
        
        # EDM forward process: x_t = x_0 + sigma_t * noise
        x_t = x0 + sigma_t * noise
        
        return x_t
    
    def denoise_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Denoising step using EDM formulation
        
        Args:
            x_t: (B, ...) noisy sample at timestep t
            noise_pred: (B, ...) predicted noise
            t: (B,) current timestep
            eta: DDIM parameter (0 = deterministic, 1 = stochastic)
        
        Returns:
            x_{t-1}: (B, ...) denoised sample
        """
        sigma_t = self.get_sigma(t)  # (B,)
        
        # Predict x0
        # In EDM: x_0 = x_t - sigma_t * noise_pred
        x0_pred = x_t - sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise_pred
        
        # Get previous timestep
        t_prev = torch.clamp(t - 1, min=0)
        sigma_t_prev = self.get_sigma(t_prev)
        
        # Reshape for broadcasting
        while len(sigma_t.shape) < len(x_t.shape):
            sigma_t = sigma_t.unsqueeze(-1)
            sigma_t_prev = sigma_t_prev.unsqueeze(-1)
        
        # DDIM-style step
        if eta > 0:
            # Stochastic sampling
            noise = torch.randn_like(x_t)
            x_prev = x0_pred + sigma_t_prev * (
                eta * noise + (1 - eta) * (x_t - x0_pred) / sigma_t
            )
        else:
            # Deterministic (DDIM)
            x_prev = x0_pred + sigma_t_prev * (x_t - x0_pred) / sigma_t
        
        return x_prev
    
    def to(self, device):
        """Move tensors to device"""
        self.sigmas = self.sigmas.to(device)
        self.sqrt_alphas_bar = self.sqrt_alphas_bar.to(device)
        self.sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.alphas_bar_prev = self.alphas_bar_prev.to(device)
        return self


class AdaptiveEDMSchedule(EDMSchedule):
    """
    Adaptive EDM schedule that adjusts based on prediction difficulty
    
    Can dynamically adjust sigma_max based on the complexity of the prediction task.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_timesteps: int = 1000,
        adaptive: bool = True
    ):
        super().__init__(sigma_min, sigma_max, rho, num_timesteps)
        self.adaptive = adaptive
        self.base_sigma_max = sigma_max
    
    def adjust_for_difficulty(self, difficulty: float):
        """
        Adjust sigma_max based on prediction difficulty
        
        Args:
            difficulty: Difficulty score [0, 1], where 1 = most difficult
        """
        if self.adaptive:
            # Higher difficulty -> higher sigma_max (more noise needed)
            self.sigma_max = self.base_sigma_max * (1.0 + 0.5 * difficulty)
            
            # Recompute schedule
            t = torch.linspace(0, 1, self.num_timesteps, dtype=torch.float32)
            sigma_max_inv_rho = self.sigma_max ** (1.0 / self.rho)
            sigma_min_inv_rho = self.sigma_min ** (1.0 / self.rho)
            
            self.sigmas = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** self.rho
            
            # Update alphas
            self.sqrt_alphas_bar = 1.0 / torch.sqrt(1.0 + self.sigmas ** 2)
            self.sqrt_one_minus_alphas_bar = self.sigmas / torch.sqrt(1.0 + self.sigmas ** 2)
            self.alphas_bar = self.sqrt_alphas_bar ** 2
            self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)

