"""
Visualization utilities for typhoon predictions
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from typing import Dict, Optional


def plot_typhoon_prediction(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    timestep: int = 0,
    output_path: Optional[str] = None
):
    """
    Visualize typhoon prediction at a specific timestep
    
    Args:
        prediction: Dictionary with 'frames', 'track', 'intensity'
        target: Dictionary with ground truth
        timestep: Which timestep to visualize
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract frames
    pred_frame = prediction['frames'][0, timestep].cpu().numpy()
    target_frame = target['frames'][0, timestep].cpu().numpy()
    
    # Plot sea level pressure
    axes[0, 0].imshow(pred_frame[30], cmap='viridis')
    axes[0, 0].set_title(f'Predicted Pressure (t={timestep})')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(target_frame[30], cmap='viridis')
    axes[1, 0].set_title(f'Target Pressure (t={timestep})')
    axes[1, 0].axis('off')
    
    # Plot wind speed (computed from u, v)
    u_pred = pred_frame[0:6].mean(axis=0)
    v_pred = pred_frame[6:12].mean(axis=0)
    wind_pred = np.sqrt(u_pred**2 + v_pred**2)
    
    u_target = target_frame[0:6].mean(axis=0)
    v_target = target_frame[6:12].mean(axis=0)
    wind_target = np.sqrt(u_target**2 + v_target**2)
    
    axes[0, 1].imshow(wind_pred, cmap='hot')
    axes[0, 1].set_title(f'Predicted Wind Speed (t={timestep})')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(wind_target, cmap='hot')
    axes[1, 1].set_title(f'Target Wind Speed (t={timestep})')
    axes[1, 1].axis('off')
    
    # Plot error
    error = np.abs(pred_frame - target_frame).mean(axis=0)
    axes[0, 2].imshow(error, cmap='Reds')
    axes[0, 2].set_title(f'Absolute Error (t={timestep})')
    axes[0, 2].axis('off')
    
    # Plot track
    pred_track = prediction['track'][0].cpu().numpy()
    target_track = target['track'][0].cpu().numpy()
    
    axes[1, 2].plot(target_track[:, 1], target_track[:, 0], 'bo-', label='Target', linewidth=2)
    axes[1, 2].plot(pred_track[:, 1], pred_track[:, 0], 'rx-', label='Predicted', linewidth=2)
    axes[1, 2].scatter(target_track[timestep, 1], target_track[timestep, 0], 
                      c='blue', s=200, marker='*', zorder=10)
    axes[1, 2].scatter(pred_track[timestep, 1], pred_track[timestep, 0], 
                      c='red', s=200, marker='*', zorder=10)
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('Latitude')
    axes[1, 2].set_title('Typhoon Track')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_track_comparison(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_path: Optional[str] = None
):
    """
    Plot multiple typhoon tracks for comparison
    
    Args:
        predictions: (B, T, 2) predicted tracks
        targets: (B, T, 2) target tracks
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    B = predictions.shape[0]
    
    for i in range(min(B, 10)):  # Plot up to 10 cases
        # Target track
        ax.plot(targets[i, :, 1], targets[i, :, 0], 
               'b-', alpha=0.3, linewidth=1)
        
        # Predicted track
        ax.plot(predictions[i, :, 1], predictions[i, :, 0], 
               'r-', alpha=0.3, linewidth=1)
        
        # Start and end points
        ax.scatter(targets[i, 0, 1], targets[i, 0, 0], 
                  c='blue', s=50, marker='o', alpha=0.5)
        ax.scatter(predictions[i, -1, 1], predictions[i, -1, 0], 
                  c='red', s=50, marker='x', alpha=0.5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='Target Track'),
        Line2D([0], [0], color='red', label='Predicted Track'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='Start'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
               markersize=8, label='End')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Typhoon Track Predictions', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_summary(
    metrics: Dict[str, float],
    output_path: Optional[str] = None
):
    """
    Plot summary of evaluation metrics
    
    Args:
        metrics: Dictionary of metrics from evaluation
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Track error vs lead time
    lead_times = [6, 12, 18, 24, 30, 36, 42, 48]
    track_errors = [metrics.get(f'track_error_{t}h_km', 0) for t in lead_times]
    
    axes[0, 0].plot(lead_times, track_errors, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Lead Time (hours)', fontsize=11)
    axes[0, 0].set_ylabel('Track Error (km)', fontsize=11)
    axes[0, 0].set_title('Track Prediction Error vs Lead Time', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Intensity error vs lead time
    intensity_errors = [metrics.get(f'intensity_mae_{t}h', 0) for t in lead_times]
    
    axes[0, 1].plot(lead_times, intensity_errors, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Lead Time (hours)', fontsize=11)
    axes[0, 1].set_ylabel('Intensity MAE (m/s)', fontsize=11)
    axes[0, 1].set_title('Intensity Prediction Error vs Lead Time', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overall metrics comparison
    metric_names = ['Structure\nMSE', 'Track Error\n(km)', 'Intensity\nMAE (m/s)']
    metric_values = [
        metrics.get('structure_mse', 0),
        metrics.get('track_error_mean_km', 0),
        metrics.get('intensity_mae', 0)
    ]
    
    axes[1, 0].bar(metric_names, metric_values, color=['blue', 'green', 'orange'])
    axes[1, 0].set_ylabel('Error', fontsize=11)
    axes[1, 0].set_title('Overall Performance Metrics', fontsize=12)
    axes[1, 0].grid(True, axis='y', alpha=0.3)
    
    # Skill scores
    skill_names = ['Frames', 'Track', 'Intensity']
    skill_values = [
        metrics.get('skill_score_frames', 0),
        metrics.get('skill_score_track', 0),
        metrics.get('skill_score_intensity', 0)
    ]
    
    axes[1, 1].bar(skill_names, skill_values, color=['purple', 'red', 'brown'])
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel('Skill Score (vs Persistence)', fontsize=11)
    axes[1, 1].set_title('Skill Scores', fontsize=12)
    axes[1, 1].set_ylim([-0.5, 1.0])
    axes[1, 1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_comparison(
    ablation_results: Dict[str, Dict],
    output_path: Optional[str] = None
):
    """
    Plot comparison of ablation study results
    
    Args:
        ablation_results: Dictionary mapping model names to metrics
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = list(ablation_results.keys())
    
    # Track error
    track_errors = [ablation_results[m].get('track_error_mean_km', 0) for m in models]
    axes[0].bar(models, track_errors, color='skyblue')
    axes[0].set_ylabel('Track Error (km)', fontsize=11)
    axes[0].set_title('Track Prediction Error', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Intensity error
    intensity_errors = [ablation_results[m].get('intensity_mae', 0) for m in models]
    axes[1].bar(models, intensity_errors, color='lightcoral')
    axes[1].set_ylabel('Intensity MAE (m/s)', fontsize=11)
    axes[1].set_title('Intensity Prediction Error', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, axis='y', alpha=0.3)
    
    # Physics validity
    physics_valid = [ablation_results[m].get('physics_valid_ratio', 0) * 100 for m in models]
    axes[2].bar(models, physics_valid, color='lightgreen')
    axes[2].set_ylabel('Physics Valid (%)', fontsize=11)
    axes[2].set_title('Physics Consistency', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylim([0, 100])
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

