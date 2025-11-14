"""
Video-to-Video Diffusion Sampler

Implements video diffusion sampling with:
1. Temporal consistency across frames
2. DDIM fast sampling
3. EDM-style noise schedule support
4. Rolling prediction support
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoDiffusionSampler:
    """
    Sampler for video-to-video diffusion models
    
    Handles:
    - Video sequence generation with temporal consistency
    - DDIM fast sampling
    - EDM and standard DDPM schedules
    - Conditional generation with past frames
    """
    
    def __init__(
        self,
        model,
        schedule,
        device: str = 'cuda',
        use_edm: bool = True
    ):
        """
        Initialize sampler
        
        Args:
            model: Diffusion model (PhysicsInformedDiffusionModel)
            schedule: Noise schedule (EDMSchedule or DiffusionSchedule)
            device: Device to run on
            use_edm: Whether using EDM schedule
        """
        self.model = model
        self.schedule = schedule
        self.device = device
        self.use_edm = use_edm
        
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        condition_dict: Dict,
        num_frames: int,
        num_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Sample video sequence using diffusion
        
        Args:
            condition_dict: Conditioning information (past_latents, etc.)
            num_frames: Number of future frames to generate
            num_steps: Number of sampling steps (None = full diffusion)
            guidance_scale: Classifier-free guidance scale
            eta: DDIM parameter (0 = deterministic, 1 = stochastic)
            return_intermediates: Whether to return intermediate steps
        
        Returns:
            Generated video latents: (B, T, C, H, W)
        """
        B = condition_dict['past_latents'].shape[0]
        C = self.model.latent_channels
        H, W = condition_dict['past_latents'].shape[3], condition_dict['past_latents'].shape[4]
        
        # Initialize with random noise
        z_t = torch.randn(B, num_frames, C, H, W, device=self.device)
        
        if num_steps is None:
            # Full diffusion sampling
            return self._full_diffusion_sample(
                z_t, condition_dict, guidance_scale, eta, return_intermediates
            )
        else:
            # DDIM fast sampling
            return self._ddim_sample(
                z_t, condition_dict, num_steps, guidance_scale, eta, return_intermediates
            )
    
    def _full_diffusion_sample(
        self,
        z_t: torch.Tensor,
        condition_dict: Dict,
        guidance_scale: float,
        eta: float,
        return_intermediates: bool
    ) -> torch.Tensor:
        """Full diffusion sampling (all timesteps)"""
        intermediates = []
        num_timesteps = self.schedule.num_timesteps
        
        for t in reversed(range(num_timesteps)):
            t_batch = torch.full((z_t.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predictions = self.model(z_t, t_batch, condition_dict, return_x0=False)
            noise_pred = predictions['noise']
            
            # Denoise step
            if self.use_edm and hasattr(self.schedule, 'denoise_step'):
                z_t = self.schedule.denoise_step(z_t, noise_pred, t_batch, eta)
            else:
                z_t = self._ddpm_step(z_t, noise_pred, t_batch, eta)
            
            if return_intermediates and t % (num_timesteps // 10) == 0:
                intermediates.append(z_t.clone())
        
        if return_intermediates:
            return z_t, intermediates
        return z_t
    
    def _ddim_sample(
        self,
        z_t: torch.Tensor,
        condition_dict: Dict,
        num_steps: int,
        guidance_scale: float,
        eta: float,
        return_intermediates: bool
    ) -> torch.Tensor:
        """DDIM fast sampling"""
        intermediates = []
        num_timesteps = self.schedule.num_timesteps
        
        # Create timestep schedule
        timesteps = torch.linspace(
            num_timesteps - 1, 0, num_steps,
            device=self.device, dtype=torch.long
        )
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((z_t.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict x0
            predictions = self.model(z_t, t_batch, condition_dict, return_x0=True)
            x0_pred = predictions.get('x0_pred', None)
            
            if x0_pred is None:
                # Fallback: predict noise and compute x0
                noise_pred = predictions['noise']
                x0_pred = self._predict_x0(z_t, noise_pred, t_batch)
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                z_t = self._ddim_step(z_t, x0_pred, t_batch, t_next, eta)
            else:
                # Final step
                z_t = x0_pred
            
            if return_intermediates:
                intermediates.append(z_t.clone())
        
        if return_intermediates:
            return z_t, intermediates
        return z_t
    
    def _predict_x0(self, z_t: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean sample x0 from noisy sample and predicted noise"""
        if self.use_edm and hasattr(self.schedule, 'get_sigma'):
            # EDM formulation: x_0 = x_t - sigma_t * noise
            sigma_t = self.schedule.get_sigma(t)
            while len(sigma_t.shape) < len(z_t.shape):
                sigma_t = sigma_t.unsqueeze(-1)
            return z_t - sigma_t * noise_pred
        else:
            # DDPM formulation
            alpha_bar_t = self.schedule.alphas_bar[t]
            while len(alpha_bar_t.shape) < len(z_t.shape):
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            return (z_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
    
    def _ddpm_step(
        self,
        z_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        eta: float
    ) -> torch.Tensor:
        """DDPM denoising step"""
        alpha_t = self.schedule.alphas[t]
        alpha_bar_t = self.schedule.alphas_bar[t]
        alpha_bar_t_prev = self.schedule.alphas_bar_prev[t]
        beta_t = self.schedule.betas[t]
        
        # Reshape for broadcasting
        while len(alpha_t.shape) < len(z_t.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # Predict x0
        pred_x0 = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        # Direction pointing to z_t
        pred_dir = torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
        
        # Random noise for stochastic sampling
        if eta > 0:
            noise = torch.randn_like(z_t)
            pred_noise = eta * torch.sqrt(beta_t) * noise
        else:
            pred_noise = 0
        
        # Previous timestep
        z_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir + pred_noise
        
        return z_prev
    
    def _ddim_step(
        self,
        z_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        eta: float
    ) -> torch.Tensor:
        """DDIM denoising step"""
        if self.use_edm and hasattr(self.schedule, 'get_sigma'):
            # EDM formulation
            sigma_t = self.schedule.get_sigma(t)
            sigma_t_next = self.schedule.get_sigma(t_next)
            
            while len(sigma_t.shape) < len(z_t.shape):
                sigma_t = sigma_t.unsqueeze(-1)
                sigma_t_next = sigma_t_next.unsqueeze(-1)
            
            # Predicted noise
            noise_pred = (z_t - x0_pred) / sigma_t
            
            # Next sample
            if eta > 0:
                noise = torch.randn_like(z_t)
                z_next = x0_pred + sigma_t_next * (eta * noise + (1 - eta) * noise_pred)
            else:
                z_next = x0_pred + sigma_t_next * noise_pred
            
            return z_next
        else:
            # Standard DDIM
            alpha_bar_t = self.schedule.alphas_bar[t]
            alpha_bar_t_next = self.schedule.alphas_bar[t_next]
            
            while len(alpha_bar_t.shape) < len(z_t.shape):
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
                alpha_bar_t_next = alpha_bar_t_next.unsqueeze(-1)
            
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_one_minus_alpha_bar_t_next = torch.sqrt(1 - alpha_bar_t_next)
            
            # Predicted noise
            pred_noise = (z_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t
            
            # Next sample
            z_next = sqrt_alpha_bar_t_next * x0_pred + sqrt_one_minus_alpha_bar_t_next * pred_noise
            
            return z_next


class RollingVideoSampler(VideoDiffusionSampler):
    """
    Rolling prediction sampler for video diffusion
    
    Implements rolling window prediction:
    - Predict N frames at a time
    - Use last M frames as new conditioning
    - Roll forward to predict longer sequences
    """
    
    def __init__(
        self,
        model,
        schedule,
        device: str = 'cuda',
        use_edm: bool = True,
        window_size: int = 12,
        overlap: int = 4
    ):
        """
        Initialize rolling sampler
        
        Args:
            model: Diffusion model
            schedule: Noise schedule
            device: Device
            use_edm: Whether using EDM
            window_size: Number of frames to predict per step
            overlap: Number of frames to overlap between windows
        """
        super().__init__(model, schedule, device, use_edm)
        self.window_size = window_size
        self.overlap = overlap
    
    @torch.no_grad()
    def rolling_sample(
        self,
        initial_condition: Dict,
        total_frames: int,
        num_steps: Optional[int] = 50,
        guidance_scale: float = 1.0,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate video using rolling prediction
        
        Args:
            initial_condition: Initial conditioning (past_latents)
            total_frames: Total number of frames to generate
            num_steps: Sampling steps per window
            guidance_scale: Guidance scale
            eta: DDIM parameter
        
        Returns:
            Generated video: (B, total_frames, C, H, W)
        """
        B = initial_condition['past_latents'].shape[0]
        C = self.model.latent_channels
        H, W = initial_condition['past_latents'].shape[3], initial_condition['past_latents'].shape[4]
        
        # Current conditioning (starts with initial past)
        current_past = initial_condition['past_latents'].clone()
        all_generated = []
        
        # Calculate number of rolling steps needed
        num_rolling_steps = (total_frames + self.window_size - self.overlap - 1) // (self.window_size - self.overlap)
        
        logger.info(f"Rolling prediction: {num_rolling_steps} steps, {total_frames} total frames")
        
        for step in range(num_rolling_steps):
            # Determine how many frames to generate in this step
            remaining = total_frames - len(all_generated)
            frames_to_generate = min(self.window_size, remaining)
            
            if frames_to_generate <= 0:
                break
            
            logger.info(f"Rolling step {step + 1}/{num_rolling_steps}: generating {frames_to_generate} frames")
            
            # Create condition dict for this step
            condition_dict = {'past_latents': current_past}
            
            # Generate frames
            generated = self.sample(
                condition_dict=condition_dict,
                num_frames=frames_to_generate,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                eta=eta
            )
            
            # Store generated frames
            all_generated.append(generated)
            
            # Update conditioning: use last overlap frames + newly generated frames
            if step < num_rolling_steps - 1:  # Not the last step
                # Take last overlap frames from current past
                past_tail = current_past[:, -self.overlap:]
                
                # Take first (window_size - overlap) frames from generated
                # (or all if this is the last window)
                gen_head = generated[:, :(self.window_size - self.overlap)]
                
                # Concatenate to form new past
                current_past = torch.cat([past_tail, gen_head], dim=1)
        
        # Concatenate all generated frames
        result = torch.cat(all_generated, dim=1)  # (B, total_frames, C, H, W)
        
        return result

