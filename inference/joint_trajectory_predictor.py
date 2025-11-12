"""
Joint Trajectory Predictor

Inference pipeline for typhoon trajectory prediction using:
1. Joint autoencoder (encodes ERA5 + IBTrACS together)
2. Diffusion model (operates on unified latent space)
3. Separate decoding (decodes to ERA5 + IBTrACS separately)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from models.autoencoder.joint_autoencoder import JointAutoencoder
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel, DiffusionSchedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointTrajectoryPredictor:
    """
    Complete inference pipeline for typhoon trajectory prediction
    
    Pipeline:
    1. Input: Past ERA5 (40, 64, 64) + Past IBTrACS (track, intensity)
    2. Encode: Joint encoder → Unified latent (8, 8, 8)
    3. Condition: Use past latents as conditioning
    4. Diffuse: Generate future latents via denoising
    5. Decode: Separate decoder → Future ERA5 + Future IBTrACS
    6. Output: Predicted trajectory, intensity, and atmospheric fields
    """
    
    def __init__(
        self,
        autoencoder: JointAutoencoder,
        diffusion_model: PhysicsInformedDiffusionModel,
        diffusion_schedule: DiffusionSchedule,
        device: str = 'cuda'
    ):
        """
        Initialize predictor
        
        Args:
            autoencoder: Trained joint autoencoder
            diffusion_model: Trained diffusion model
            diffusion_schedule: Diffusion noise schedule
            device: Device to run on
        """
        self.autoencoder = autoencoder.to(device)
        self.diffusion_model = diffusion_model.to(device)
        self.diffusion_schedule = diffusion_schedule.to(device)
        self.device = device
        
        # Set to eval mode
        self.autoencoder.eval()
        self.diffusion_model.eval()
        
    @torch.no_grad()
    def predict(
        self,
        past_frames: torch.Tensor,
        past_track: torch.Tensor,
        past_intensity: torch.Tensor,
        num_future_steps: int = 12,
        num_samples: int = 1,
        ddim_steps: Optional[int] = 50,
        guidance_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future typhoon trajectory and intensity
        
        Args:
            past_frames: (B, T_past, C, H, W) - past ERA5 atmospheric fields
            past_track: (B, T_past, 2) - past track coordinates (lat, lon)
            past_intensity: (B, T_past,) - past wind speeds
            num_future_steps: Number of future timesteps to predict
            num_samples: Number of samples to generate (for uncertainty)
            ddim_steps: Number of DDIM sampling steps (None = full diffusion)
            guidance_scale: Classifier-free guidance scale
        
        Returns:
            Dictionary containing:
                - 'future_frames': (B, num_samples, T_future, C, H, W) predicted ERA5
                - 'future_track': (B, num_samples, T_future, 2) predicted trajectory
                - 'future_intensity': (B, num_samples, T_future) predicted intensity
                - 'latents': (B, num_samples, T_future, C_latent, H/8, W/8) latent codes
        """
        B, T_past, C, H, W = past_frames.shape
        
        # ═════════════════════════════════════════════════════════════
        # STEP 1: Joint Encoding (ERA5 + IBTrACS → Unified Latent)
        # ═════════════════════════════════════════════════════════════
        
        logger.info("Encoding past observations...")
        
        # Flatten temporal dimension
        past_frames_flat = past_frames.reshape(B * T_past, C, H, W)
        past_track_flat = past_track.reshape(B * T_past, 2)
        past_intensity_flat = past_intensity.reshape(B * T_past)
        
        # Joint encode
        past_latents = self.autoencoder.encode(
            past_frames_flat,
            past_track_flat,
            past_intensity_flat
        )
        
        # Reshape back
        C_latent = past_latents.shape[1]
        H_latent, W_latent = past_latents.shape[2], past_latents.shape[3]
        past_latents = past_latents.reshape(B, T_past, C_latent, H_latent, W_latent)
        
        # ═════════════════════════════════════════════════════════════
        # STEP 2: Prepare Conditioning
        # ═════════════════════════════════════════════════════════════
        
        condition_dict = {
            'past_latents': past_latents,
            # No need for separate track/intensity - already in latent!
        }
        
        # ═════════════════════════════════════════════════════════════
        # STEP 3: Generate Future Latents via Diffusion
        # ═════════════════════════════════════════════════════════════
        
        logger.info(f"Generating {num_samples} future predictions...")
        
        all_future_latents = []
        all_future_tracks = []
        all_future_intensities = []
        all_future_frames = []
        
        for sample_idx in range(num_samples):
            # Initialize with random noise
            future_latents_shape = (B, num_future_steps, C_latent, H_latent, W_latent)
            z_t = torch.randn(future_latents_shape, device=self.device)
            
            # Denoising loop
            if ddim_steps is not None:
                # DDIM sampling (faster)
                future_latents = self._ddim_sample(
                    z_t, condition_dict, ddim_steps, guidance_scale
                )
            else:
                # Full diffusion sampling
                future_latents = self._diffusion_sample(
                    z_t, condition_dict, guidance_scale
                )
            
            all_future_latents.append(future_latents)
            
            # ═════════════════════════════════════════════════════════════
            # STEP 4: Decode Latents Separately (Latent → ERA5 + IBTrACS)
            # ═════════════════════════════════════════════════════════════
            
            logger.info(f"Decoding sample {sample_idx + 1}/{num_samples}...")
            
            # Flatten for decoding
            future_latents_flat = future_latents.reshape(
                B * num_future_steps, C_latent, H_latent, W_latent
            )
            
            # Decode separately
            future_frames_flat, future_track_flat, future_intensity_flat = \
                self.autoencoder.decode(future_latents_flat)
            
            # Reshape back
            future_frames = future_frames_flat.reshape(B, num_future_steps, C, H, W)
            future_track = future_track_flat.reshape(B, num_future_steps, 2)
            future_intensity = future_intensity_flat.reshape(B, num_future_steps)
            
            all_future_frames.append(future_frames)
            all_future_tracks.append(future_track)
            all_future_intensities.append(future_intensity)
        
        # Stack samples
        results = {
            'future_frames': torch.stack(all_future_frames, dim=1),  # (B, N_samples, T, C, H, W)
            'future_track': torch.stack(all_future_tracks, dim=1),    # (B, N_samples, T, 2)
            'future_intensity': torch.stack(all_future_intensities, dim=1),  # (B, N_samples, T)
            'latents': torch.stack(all_future_latents, dim=1)  # (B, N_samples, T, C_l, H_l, W_l)
        }
        
        logger.info("Prediction complete!")
        
        return results
    
    def _diffusion_sample(
        self,
        z_t: torch.Tensor,
        condition_dict: Dict,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Full diffusion sampling (all timesteps)
        
        Args:
            z_t: Initial noise
            condition_dict: Conditioning information
            guidance_scale: Guidance scale
        
        Returns:
            Denoised latent
        """
        B = z_t.shape[0]
        
        # Reverse diffusion process
        for t in reversed(range(self.diffusion_schedule.num_timesteps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predictions = self.diffusion_model(z_t, t_batch, condition_dict, return_x0=False)
            noise_pred = predictions['noise']
            
            # Denoise step
            z_t = self.diffusion_schedule.denoise_step(z_t, noise_pred, t_batch)
        
        return z_t
    
    def _ddim_sample(
        self,
        z_t: torch.Tensor,
        condition_dict: Dict,
        num_steps: int = 50,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        DDIM sampling (faster, deterministic)
        
        Args:
            z_t: Initial noise
            condition_dict: Conditioning information
            num_steps: Number of sampling steps
            guidance_scale: Guidance scale
        
        Returns:
            Denoised latent
        """
        B = z_t.shape[0]
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.diffusion_schedule.num_timesteps - 1, 0, num_steps,
            device=self.device, dtype=torch.long
        )
        
        # DDIM sampling loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            # Predict x0
            predictions = self.diffusion_model(z_t, t_batch, condition_dict, return_x0=True)
            x0_pred = predictions.get('x0_pred', None)
            
            if x0_pred is None:
                # Fallback: predict noise and compute x0
                noise_pred = predictions['noise']
                alpha_t = self.diffusion_schedule.alphas_bar[t]
                x0_pred = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.diffusion_schedule.alphas_bar[t]
                alpha_t_next = self.diffusion_schedule.alphas_bar[t_next]
                
                # Predicted noise
                noise_pred = (z_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t)
                
                # Next sample
                z_t = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
            else:
                # Final step
                z_t = x0_pred
        
        return z_t
    
    def predict_trajectory(
        self,
        past_frames: torch.Tensor,
        past_track: torch.Tensor,
        past_intensity: torch.Tensor,
        num_future_steps: int = 12,
        num_samples: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Predict trajectory with uncertainty quantification
        
        Args:
            past_frames: Past atmospheric fields
            past_track: Past trajectory
            past_intensity: Past intensity
            num_future_steps: Number of future steps
            num_samples: Number of ensemble samples
        
        Returns:
            Dictionary with predictions and uncertainty:
                - 'track_mean': (T_future, 2) mean trajectory
                - 'track_std': (T_future, 2) trajectory uncertainty
                - 'intensity_mean': (T_future,) mean intensity
                - 'intensity_std': (T_future,) intensity uncertainty
                - 'samples': all ensemble samples
        """
        # Generate predictions
        results = self.predict(
            past_frames, past_track, past_intensity,
            num_future_steps=num_future_steps,
            num_samples=num_samples,
            ddim_steps=50
        )
        
        # Extract trajectory and intensity
        future_track = results['future_track']  # (B, N_samples, T, 2) - NORMALIZED
        future_intensity = results['future_intensity']  # (B, N_samples, T) - NORMALIZED
        
        # ═════════════════════════════════════════════════════════════
        # DENORMALIZE: Convert from normalized space to physical units
        # ═════════════════════════════════════════════════════════════
        # Track normalization: lat = (lat_norm * 12.5) + 22.5, lon = (lon_norm * 20.0) + 140.0
        # Intensity normalization: wind = (wind_norm * 26.5) + 43.5
        # These are the inverse of the normalization used during training
        
        # Denormalize track (lat, lon)
        future_track_denorm = future_track.clone()
        future_track_denorm[:, :, :, 0] = future_track[:, :, :, 0] * 12.5 + 22.5  # Latitude: WP range 10-35°N
        future_track_denorm[:, :, :, 1] = future_track[:, :, :, 1] * 20.0 + 140.0  # Longitude: WP range 120-160°E
        
        # Denormalize intensity (wind speed)
        future_intensity_denorm = future_intensity * 26.5 + 43.5  # Wind speed: typical range 17-70 m/s
        
        # Compute statistics (assume B=1 for single case)
        track_samples = future_track_denorm[0].cpu().numpy()  # (N_samples, T, 2) - PHYSICAL UNITS
        intensity_samples = future_intensity_denorm[0].cpu().numpy()  # (N_samples, T) - PHYSICAL UNITS
        
        track_mean = track_samples.mean(axis=0)  # (T, 2)
        track_std = track_samples.std(axis=0)    # (T, 2)
        intensity_mean = intensity_samples.mean(axis=0)  # (T,)
        intensity_std = intensity_samples.std(axis=0)    # (T,)
        
        return {
            'track_mean': track_mean,
            'track_std': track_std,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'track_samples': track_samples,
            'intensity_samples': intensity_samples,
            'frames_samples': results['future_frames'][0].cpu().numpy()
        }
    
    @classmethod
    def from_checkpoints(
        cls,
        autoencoder_path: str,
        diffusion_path: str,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Load predictor from checkpoint files
        
        Args:
            autoencoder_path: Path to joint autoencoder checkpoint
            diffusion_path: Path to diffusion model checkpoint
            config: Configuration dictionary
            device: Device to run on
        
        Returns:
            Initialized predictor
        """
        logger.info("Loading models from checkpoints...")
        
        # Load autoencoder
        autoencoder = JointAutoencoder(
            era5_channels=config.get('era5_channels', 40),
            latent_channels=config.get('latent_channels', 8)
        )
        ae_checkpoint = torch.load(autoencoder_path, map_location=device)
        autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
        
        # Load diffusion model
        diffusion_model = PhysicsInformedDiffusionModel(
            latent_channels=config.get('latent_channels', 8),
            hidden_dim=config.get('hidden_dim', 256),
            num_heads=config.get('num_heads', 8),
            output_frames=config.get('output_frames', 12)
        )
        diff_checkpoint = torch.load(diffusion_path, map_location=device)
        
        # Use EMA model if available
        if 'ema_model_state_dict' in diff_checkpoint:
            diffusion_model.load_state_dict(diff_checkpoint['ema_model_state_dict'])
        else:
            diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
        
        # Create diffusion schedule
        diffusion_schedule = DiffusionSchedule(
            num_timesteps=config.get('timesteps', 1000),
            beta_start=config.get('beta_start', 1e-4),
            beta_end=config.get('beta_end', 0.02)
        )
        
        logger.info("Models loaded successfully!")
        
        return cls(autoencoder, diffusion_model, diffusion_schedule, device)

