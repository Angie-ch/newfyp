"""
Multi-Task Diffusion Loss

Combines diffusion loss, track loss, intensity loss, physics losses, and consistency losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.physics_constraints import PhysicsLossComputer


class MultiTaskDiffusionLoss(nn.Module):
    """
    Combined loss with physics constraints and multi-task objectives
    """
    
    def __init__(
        self,
        autoencoder=None,
        diffusion_weight: float = 1.0,
        track_weight: float = 0.5,
        intensity_weight: float = 0.3,
        physics_weight: float = 0.2,
        consistency_weight: float = 0.1,
        physics_weights: dict = None
    ):
        """
        Initialize multi-task loss
        
        Args:
            autoencoder: Trained autoencoder for physics loss computation
            diffusion_weight: Weight for diffusion (noise prediction) loss
            track_weight: Weight for track prediction loss
            intensity_weight: Weight for intensity prediction loss
            physics_weight: Weight for physics constraint losses
            consistency_weight: Weight for cross-task consistency losses
            physics_weights: Individual weights for physics loss components
        """
        super().__init__()
        
        self.autoencoder = autoencoder
        
        # Loss weights
        self.w_diffusion = diffusion_weight
        self.w_track = track_weight
        self.w_intensity = intensity_weight
        self.w_physics = physics_weight
        self.w_consistency = consistency_weight
        
        # Physics loss weights
        if physics_weights is None:
            physics_weights = {
                'geostrophic': 1.0,
                'mass_conservation': 0.1,
                'temporal_smooth': 0.1,
                'wind_pressure': 0.5
            }
        self.physics_weights = physics_weights
        
        # Physics loss computer
        self.physics_computer = PhysicsLossComputer(autoencoder)
    
    def forward(self, predictions, targets):
        """
        Compute total loss
        
        Args:
            predictions: {
                'noise': (B, T, C_l, H/8, W/8),
                'track': (B, T, 2),
                'intensity': (B, T),
                'x0_pred': (B, T, C_l, H/8, W/8)
            }
            targets: {
                'noise': ground truth noise,
                'track': (B, T, 2) true coordinates,
                'intensity': (B, T) true wind speeds,
                'frames': (B, T, C_l, H/8, W/8) clean latents
            }
        
        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # 1. Diffusion loss (main objective)
        loss_diffusion = F.mse_loss(
            predictions['noise'],
            targets['noise']
        )
        loss_dict['diffusion'] = loss_diffusion.item()
        
        # 2. Track loss
        loss_track = F.mse_loss(
            predictions['track'],
            targets['track']
        )
        loss_dict['track'] = loss_track.item()
        
        # 3. Intensity loss
        loss_intensity = F.mse_loss(
            predictions['intensity'],
            targets['intensity']
        )
        loss_dict['intensity'] = loss_intensity.item()
        
        # 4. Physics losses
        if 'x0_pred' in predictions and self.w_physics > 0:
            physics_losses = self.physics_computer(
                predictions['x0_pred'],
                self.autoencoder
            )
            
            # Compute weighted sum of physics losses (only for keys that exist)
            loss_physics = torch.tensor(0.0, device=predictions['noise'].device, dtype=torch.float32)
            for key, weight in self.physics_weights.items():
                if key in physics_losses:
                    loss_physics += weight * physics_losses[key]
                    loss_dict[f'physics_{key}'] = physics_losses[key].item()
            
            loss_dict['physics_total'] = loss_physics.item()
        else:
            loss_physics = torch.tensor(0.0, device=predictions['noise'].device, dtype=torch.float32)
            loss_dict['physics_total'] = 0.0
        
        # 5. Cross-task consistency
        if 'x0_pred' in predictions and self.w_consistency > 0:
            loss_consistency = self.compute_consistency_loss(
                predictions['x0_pred'],
                predictions['track'],
                predictions['intensity']
            )
            loss_dict['consistency'] = loss_consistency.item()
        else:
            loss_consistency = torch.tensor(0.0, device=predictions['noise'].device, dtype=torch.float32)
            loss_dict['consistency'] = 0.0
        
        # Total loss
        total_loss = (
            self.w_diffusion * loss_diffusion +
            self.w_track * loss_track +
            self.w_intensity * loss_intensity +
            self.w_physics * loss_physics +
            self.w_consistency * loss_consistency
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_consistency_loss(self, structure, track, intensity):
        """
        Ensure track/intensity match structure predictions
        
        This encourages the model to make physically consistent predictions
        across different tasks
        
        Args:
            structure: (B, T, C_l, H, W) predicted latent structure
            track: (B, T, 2) predicted track
            intensity: (B, T) predicted intensity
        
        Returns:
            consistency_loss: Scalar
        """
        B, T, C_l, H, W = structure.shape
        
        # For consistency, we check if the spatial location of maximum
        # intensity in the structure aligns with the predicted track
        
        # Find location of maximum magnitude in structure (proxy for center)
        structure_magnitude = structure.abs().sum(dim=2)  # (B, T, H, W)
        structure_flat = structure_magnitude.flatten(2)  # (B, T, H*W)
        
        max_indices = structure_flat.argmax(dim=2)  # (B, T)
        
        predicted_centers_y = (max_indices // W).float() / H
        predicted_centers_x = (max_indices % W).float() / W
        
        predicted_centers = torch.stack([
            predicted_centers_y,
            predicted_centers_x
        ], dim=-1)  # (B, T, 2)
        
        # Normalize track to [0, 1] range if needed
        # (Assume track is already in normalized coordinates or we normalize it)
        track_normalized = track  # Adjust based on actual track representation
        
        # Consistency loss: centers should match
        loss_center = F.mse_loss(predicted_centers, track_normalized)
        
        # Additional: intensity should correlate with structure magnitude
        structure_max = structure_flat.max(dim=2)[0]  # (B, T)
        
        # Normalize both to similar scales
        structure_max_norm = (structure_max - structure_max.mean()) / (structure_max.std() + 1e-8)
        intensity_norm = (intensity - intensity.mean()) / (intensity.std() + 1e-8)
        
        loss_intensity_match = F.mse_loss(structure_max_norm, intensity_norm)
        
        return loss_center + 0.5 * loss_intensity_match


class WeightedMSELoss(nn.Module):
    """
    MSE loss with optional time-dependent weighting
    """
    
    def __init__(self, time_weights=None):
        """
        Args:
            time_weights: (T,) tensor of weights for each timestep
        """
        super().__init__()
        self.time_weights = time_weights
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, ...) predictions
            target: (B, T, ...) targets
        
        Returns:
            Weighted MSE loss
        """
        mse = (pred - target) ** 2
        
        if self.time_weights is not None:
            # Apply time-dependent weights
            weights = self.time_weights.to(pred.device)
            # Reshape for broadcasting
            while len(weights.shape) < len(mse.shape):
                weights = weights.unsqueeze(0).unsqueeze(-1)
            
            mse = mse * weights
        
        return mse.mean()


class HuberLoss(nn.Module):
    """
    Huber loss (smooth L1 loss) - robust to outliers
    """
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        """
        Args:
            pred: predictions
            target: targets
        
        Returns:
            Huber loss
        """
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        return loss.mean()

