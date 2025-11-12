"""
Prediction Evaluation Metrics

Comprehensive metrics for typhoon prediction evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


class PredictionMetrics:
    """
    Compute various metrics for typhoon predictions
    """
    
    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error"""
        return F.l1_loss(pred, target).item()
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Root Mean Squared Error"""
        return torch.sqrt(F.mse_loss(pred, target)).item()
    
    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """
        Structural Similarity Index (SSIM)
        
        Args:
            pred: (B, C, H, W) predictions
            target: (B, C, H, W) targets
            window_size: Size of Gaussian window
        
        Returns:
            SSIM score (higher is better, max = 1.0)
        """
        # Simplified SSIM computation
        # Full implementation would use Gaussian window
        
        mu_pred = pred.mean()
        mu_target = target.mean()
        
        sigma_pred = pred.std()
        sigma_target = target.std()
        
        sigma_pred_target = ((pred - mu_pred) * (target - mu_target)).mean()
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred**2 + sigma_target**2 + C2))
        
        return ssim.item()
    
    @staticmethod
    def track_error(
        pred_track: torch.Tensor,
        target_track: torch.Tensor,
        degrees_to_km: float = 111.0
    ) -> Dict[str, float]:
        """
        Track prediction error
        
        Args:
            pred_track: (B, T, 2) predicted (lat, lon)
            target_track: (B, T, 2) target (lat, lon)
            degrees_to_km: Conversion factor from degrees to km
        
        Returns:
            Dictionary of track metrics
        """
        # Euclidean distance in degrees
        error_deg = torch.norm(pred_track - target_track, dim=-1)  # (B, T)
        
        # Convert to km
        error_km = error_deg * degrees_to_km
        
        metrics = {
            'track_error_mean_km': error_km.mean().item(),
            'track_error_std_km': error_km.std().item(),
            'track_error_max_km': error_km.max().item(),
            'track_error_min_km': error_km.min().item()
        }
        
        # Per-timestep errors
        T = pred_track.shape[1]
        for t in range(T):
            lead_time_hours = (t + 1) * 6
            metrics[f'track_error_{lead_time_hours}h_km'] = error_km[:, t].mean().item()
        
        return metrics
    
    @staticmethod
    def intensity_error(
        pred_intensity: torch.Tensor,
        target_intensity: torch.Tensor
    ) -> Dict[str, float]:
        """
        Intensity prediction error
        
        Args:
            pred_intensity: (B, T) predicted wind speeds
            target_intensity: (B, T) target wind speeds
        
        Returns:
            Dictionary of intensity metrics
        """
        error = torch.abs(pred_intensity - target_intensity)  # (B, T)
        
        metrics = {
            'intensity_mae': error.mean().item(),
            'intensity_rmse': torch.sqrt((error ** 2).mean()).item(),
            'intensity_max_error': error.max().item()
        }
        
        # Per-timestep errors
        T = pred_intensity.shape[1]
        for t in range(T):
            lead_time_hours = (t + 1) * 6
            metrics[f'intensity_mae_{lead_time_hours}h'] = error[:, t].mean().item()
        
        # Bias (mean signed error)
        bias = (pred_intensity - target_intensity).mean().item()
        metrics['intensity_bias'] = bias
        
        return metrics
    
    @staticmethod
    def skill_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        baseline_pred: torch.Tensor
    ) -> float:
        """
        Skill score comparing model to baseline
        
        SS = 1 - (MSE_model / MSE_baseline)
        
        Args:
            pred: Model predictions
            target: Ground truth
            baseline_pred: Baseline predictions
        
        Returns:
            Skill score (higher is better, 0 = baseline, 1 = perfect)
        """
        mse_model = F.mse_loss(pred, target)
        mse_baseline = F.mse_loss(baseline_pred, target)
        
        skill = 1 - (mse_model / mse_baseline)
        
        return skill.item()


def compute_all_metrics(predictions: Dict, targets: Dict) -> Dict:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        predictions: {
            'frames': (B, T, C, H, W),
            'track': (B, T, 2),
            'intensity': (B, T)
        }
        targets: Same structure as predictions
    
    Returns:
        Dictionary of all metrics
    """
    metrics = PredictionMetrics()
    all_metrics = {}
    
    # Structure metrics (pixel-level)
    all_metrics['structure_mse'] = metrics.mse(
        predictions['frames'],
        targets['frames']
    )
    
    all_metrics['structure_mae'] = metrics.mae(
        predictions['frames'],
        targets['frames']
    )
    
    all_metrics['structure_rmse'] = metrics.rmse(
        predictions['frames'],
        targets['frames']
    )
    
    # SSIM per timestep
    B, T, C, H, W = predictions['frames'].shape
    ssim_scores = []
    for t in range(T):
        ssim_t = metrics.ssim(
            predictions['frames'][:, t],
            targets['frames'][:, t]
        )
        ssim_scores.append(ssim_t)
    
    all_metrics['structure_ssim_mean'] = np.mean(ssim_scores)
    all_metrics['structure_ssim_std'] = np.std(ssim_scores)
    
    # Track metrics
    track_metrics = metrics.track_error(
        predictions['track'],
        targets['track']
    )
    all_metrics.update(track_metrics)
    
    # Intensity metrics
    intensity_metrics = metrics.intensity_error(
        predictions['intensity'],
        targets['intensity']
    )
    all_metrics.update(intensity_metrics)
    
    return all_metrics


def compute_baseline_metrics(predictions: Dict, targets: Dict, baseline_type: str = 'persistence') -> Dict:
    """
    Compute metrics for baseline methods
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        baseline_type: 'persistence' or 'linear'
    
    Returns:
        Dictionary of baseline comparison metrics
    """
    if baseline_type == 'persistence':
        # Last frame repeated
        baseline_frames = targets['frames'][:, :1].repeat(1, targets['frames'].shape[1], 1, 1, 1)
        baseline_track = targets['track'][:, :1].repeat(1, targets['track'].shape[1], 1)
        baseline_intensity = targets['intensity'][:, :1].repeat(1, targets['intensity'].shape[1])
    elif baseline_type == 'linear':
        # Linear extrapolation (simplified)
        # Would need actual implementation based on past frames
        baseline_frames = targets['frames']  # Placeholder
        baseline_track = targets['track']  # Placeholder
        baseline_intensity = targets['intensity']  # Placeholder
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    baseline_preds = {
        'frames': baseline_frames,
        'track': baseline_track,
        'intensity': baseline_intensity
    }
    
    # Compute metrics for baseline
    baseline_metrics = compute_all_metrics(baseline_preds, targets)
    
    # Add prefix
    baseline_metrics = {f'baseline_{k}': v for k, v in baseline_metrics.items()}
    
    # Compute skill scores
    metrics = PredictionMetrics()
    
    skill_frames = metrics.skill_score(
        predictions['frames'],
        targets['frames'],
        baseline_frames
    )
    
    skill_track = metrics.skill_score(
        predictions['track'],
        targets['track'],
        baseline_track
    )
    
    skill_intensity = metrics.skill_score(
        predictions['intensity'],
        targets['intensity'],
        baseline_intensity
    )
    
    baseline_metrics.update({
        'skill_score_frames': skill_frames,
        'skill_score_track': skill_track,
        'skill_score_intensity': skill_intensity
    })
    
    return baseline_metrics

