"""
Physics Constraint Layers

Enforce atmospheric physics laws on predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsProjector(nn.Module):
    """
    Projects predictions onto physically consistent manifold
    
    Uses a learned correction network to adjust predictions
    toward physics-consistent states
    """
    
    def __init__(self, latent_channels: int = 8):
        super().__init__()
        
        # Learned correction network
        self.correction_net = nn.Sequential(
            nn.Conv3d(latent_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv3d(64, latent_channels, 3, padding=1)
        )
        
        # Learnable strength parameter
        self.correction_strength = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    
    def forward(self, x0_pred):
        """
        Enforce physics constraints on predicted latents
        
        Args:
            x0_pred: (B, T, C_latent, H/8, W/8) predicted clean state
        
        Returns:
            x0_constrained: Same shape, but physics-consistent
        """
        # Rearrange to (B, C, T, H, W)
        x = x0_pred.permute(0, 2, 1, 3, 4)
        
        # Compute physics residual
        physics_residual = self.correction_net(x)
        
        # Apply correction with learnable strength
        x_corrected = x + self.correction_strength * physics_residual
        
        # Rearrange back
        x0_constrained = x_corrected.permute(0, 2, 1, 3, 4)
        
        return x0_constrained


class PhysicsLossComputer:
    """
    Compute physics-based loss terms
    
    Checks for:
    1. Geostrophic balance
    2. Mass conservation
    3. Temporal smoothness
    4. Wind-pressure relationship
    """
    
    def __init__(self, autoencoder=None, resolution: float = 0.25):
        """
        Args:
            autoencoder: Trained autoencoder to decode latents
            resolution: Spatial resolution in degrees
        """
        self.autoencoder = autoencoder
        self.resolution = resolution
        
        # Coriolis parameter (approximate at 20°N latitude)
        self.f = 5e-5  # rad/s
    
    def __call__(self, pred_latents, autoencoder=None):
        """
        Compute all physics losses
        
        Args:
            pred_latents: (B, T, C_latent, H, W) in latent space
            autoencoder: Optional autoencoder override
        
        Returns:
            Dictionary of physics loss terms
        """
        if autoencoder is None:
            autoencoder = self.autoencoder
        
        if autoencoder is None:
            # Return dummy losses if no autoencoder
            return {
                'geostrophic': torch.tensor(0.0, device=pred_latents.device, dtype=torch.float32),
                'mass_conservation': torch.tensor(0.0, device=pred_latents.device, dtype=torch.float32),
                'temporal_smooth': torch.tensor(0.0, device=pred_latents.device, dtype=torch.float32),
                'wind_pressure': torch.tensor(0.0, device=pred_latents.device, dtype=torch.float32)
            }
        
        # Decode to physical space
        B, T, C_l, H_l, W_l = pred_latents.shape
        pred_flat = pred_latents.reshape(B * T, C_l, H_l, W_l)
        
        with torch.no_grad():
            decoded = autoencoder.decode(pred_flat)  # (B*T, C, H, W)
        
        C, H, W = decoded.shape[1:]
        decoded = decoded.reshape(B, T, C, H, W)
        
        # Extract variables (assuming channel order)
        # Channels 0-5: u-wind at 6 pressure levels
        # Channels 6-11: v-wind at 6 pressure levels
        # Channel 30: sea level pressure (index may vary)
        
        u = decoded[:, :, 0:6, :, :]   # (B, T, 6, H, W)
        v = decoded[:, :, 6:12, :, :]  # (B, T, 6, H, W)
        
        # Assume pressure is around channel 30 (adjust based on actual data)
        if C > 30:
            p = decoded[:, :, 30, :, :]  # (B, T, H, W)
        else:
            # Use dummy pressure if not available
            p = torch.zeros(B, T, H, W, device=decoded.device, dtype=torch.float32)
        
        # Compute losses
        losses = {}
        
        # 1. Geostrophic balance (at 700 hPa, level 3)
        losses['geostrophic'] = self._compute_geostrophic_loss(
            u[:, :, 3, :, :],
            v[:, :, 3, :, :],
            p
        )
        
        # 2. Mass conservation (divergence ≈ 0)
        losses['mass_conservation'] = self._compute_mass_conservation_loss(u, v)
        
        # 3. Temporal smoothness
        losses['temporal_smooth'] = self._compute_temporal_smoothness(pred_latents)
        
        # 4. Wind-pressure relationship
        losses['wind_pressure'] = self._compute_wind_pressure_consistency(u, v, p)
        
        return losses
    
    def _compute_geostrophic_loss(self, u, v, p):
        """
        Geostrophic balance: f × u_geo = -∂p/∂y, f × v_geo = ∂p/∂x
        """
        # Compute pressure gradients
        dp_dx = self._gradient_x(p)
        dp_dy = self._gradient_y(p)
        
        # Geostrophic wind from pressure
        u_geo = -(1 / self.f) * dp_dy
        v_geo = (1 / self.f) * dp_dx
        
        # Loss: difference between actual and geostrophic wind
        loss = F.mse_loss(u, u_geo) + F.mse_loss(v, v_geo)
        
        return loss
    
    def _compute_mass_conservation_loss(self, u, v):
        """
        Mass conservation: ∂u/∂x + ∂v/∂y ≈ 0
        """
        # Compute divergence
        du_dx = self._gradient_x(u)
        dv_dy = self._gradient_y(v)
        divergence = du_dx + dv_dy
        
        # Loss: divergence should be small
        loss = torch.mean(divergence ** 2)
        
        return loss
    
    def _compute_temporal_smoothness(self, x):
        """
        Temporal smoothness: consecutive frames should change smoothly
        """
        # Compute frame differences
        frame_diff = x[:, 1:] - x[:, :-1]
        
        # Loss: differences should be small
        loss = torch.mean(frame_diff ** 2)
        
        return loss
    
    def _compute_wind_pressure_consistency(self, u, v, p):
        """
        Wind-pressure relationship: stronger winds → lower pressure
        Empirical: P_min ≈ 1013 - k * V_max
        """
        # Compute wind speed
        wind_speed = torch.sqrt(u**2 + v**2)
        
        # Maximum wind speed (over levels and space)
        wind_max = wind_speed.max(dim=2)[0]  # Max over levels
        wind_max = wind_max.flatten(2).max(dim=2)[0]  # Max over space: (B, T)
        
        # Minimum pressure (over space)
        pressure_min = p.flatten(2).min(dim=2)[0]  # (B, T)
        
        # Empirical relationship (adjust coefficient as needed)
        expected_pressure = 1013.0 - 0.5 * wind_max
        
        # Loss
        loss = F.mse_loss(pressure_min, expected_pressure)
        
        return loss
    
    def _gradient_x(self, field):
        """Compute ∂/∂x using central finite differences"""
        # For 5D tensors (B, T, C, H, W), flatten to 4D for padding
        original_shape = field.shape
        
        if len(original_shape) == 5:
            B, T, C, H, W = original_shape
            field = field.reshape(B * T * C, H, W)
        elif len(original_shape) == 4:
            B, T, H, W = original_shape
            field = field.reshape(B * T, H, W)
        elif len(original_shape) > 5:
            # Merge to 3D
            field = field.reshape(-1, original_shape[-2], original_shape[-1])
        
        # Pad for boundary (now working with 3D: B, H, W)
        padded = F.pad(field, (1, 1, 0, 0), mode='replicate')
        
        # Central difference
        grad = (padded[..., :, 2:] - padded[..., :, :-2]) / (2 * self.resolution)
        
        # Restore original shape
        grad = grad.reshape(*original_shape)
        
        return grad
    
    def _gradient_y(self, field):
        """Compute ∂/∂y using central finite differences"""
        # For 5D tensors (B, T, C, H, W), flatten to 3D for padding
        original_shape = field.shape
        
        if len(original_shape) == 5:
            B, T, C, H, W = original_shape
            field = field.reshape(B * T * C, H, W)
        elif len(original_shape) == 4:
            B, T, H, W = original_shape
            field = field.reshape(B * T, H, W)
        elif len(original_shape) > 5:
            # Merge to 3D
            field = field.reshape(-1, original_shape[-2], original_shape[-1])
        
        # Pad for boundary (now working with 3D: B, H, W)
        padded = F.pad(field, (0, 0, 1, 1), mode='replicate')
        
        # Central difference
        grad = (padded[..., 2:, :] - padded[..., :-2, :]) / (2 * self.resolution)
        
        # Restore original shape
        grad = grad.reshape(*original_shape)
        
        return grad


def validate_physics(frames, threshold_dict=None):
    """
    Validate if predictions satisfy basic physics constraints
    
    Args:
        frames: (B, T, C, H, W) atmospheric fields in physical units
        threshold_dict: Dictionary of validation thresholds
    
    Returns:
        Dictionary of validation results
    """
    if threshold_dict is None:
        threshold_dict = {
            'wind_max': 100.0,      # m/s
            'wind_min': 0.0,        # m/s
            'pressure_max': 1020.0, # hPa
            'pressure_min': 850.0,  # hPa
            'temporal_jump': 10.0   # Max change between frames
        }
    
    checks = {}
    
    # Extract wind components
    u = frames[:, :, 0:6, :, :]
    v = frames[:, :, 6:12, :, :]
    wind_speed = torch.sqrt(u**2 + v**2)
    
    # Check wind range
    wind_max = wind_speed.max().item()
    wind_min = wind_speed.min().item()
    
    checks['wind_max'] = wind_max
    checks['wind_min'] = wind_min
    checks['wind_range_ok'] = (wind_min >= threshold_dict['wind_min'] and 
                                wind_max <= threshold_dict['wind_max'])
    
    # Check pressure range (if available)
    if frames.shape[2] > 30:
        pressure = frames[:, :, 30, :, :]
        pressure_max = pressure.max().item()
        pressure_min = pressure.min().item()
        
        checks['pressure_max'] = pressure_max
        checks['pressure_min'] = pressure_min
        checks['pressure_range_ok'] = (pressure_min >= threshold_dict['pressure_min'] and
                                        pressure_max <= threshold_dict['pressure_max'])
    else:
        checks['pressure_range_ok'] = True
    
    # Check temporal smoothness
    frame_diffs = torch.diff(frames, dim=1)
    temporal_jump = frame_diffs.abs().max().item()
    
    checks['temporal_jump'] = temporal_jump
    checks['temporal_ok'] = temporal_jump < threshold_dict['temporal_jump']
    
    # Overall validity
    checks['is_valid'] = all([
        checks['wind_range_ok'],
        checks['pressure_range_ok'],
        checks['temporal_ok']
    ])
    
    return checks

