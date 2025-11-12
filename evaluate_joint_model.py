"""
Evaluation Script for Joint ERA5 + IBTrACS Model

Evaluates the complete pipeline on test data and generates predictions
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.datasets.typhoon_dataset import TyphoonDataset
from torch.utils.data import DataLoader
from inference.joint_trajectory_predictor import JointTrajectoryPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(
    predictor: JointTrajectoryPredictor,
    test_loader: DataLoader,
    num_samples: int = 10,
    save_dir: Path = None
):
    """
    Evaluate model on test set
    
    Args:
        predictor: Trained predictor
        test_loader: Test data loader
        num_samples: Number of ensemble samples
        save_dir: Directory to save results
    """
    all_track_errors = []
    all_intensity_errors = []
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating on {len(test_loader)} batches...")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        # Get data
        past_frames = batch['past_frames'].cuda()
        past_track = batch['track_past'].cuda()
        past_intensity = batch['intensity_past'].cuda()
        
        target_track = batch['track_future'].cpu().numpy()
        target_intensity = batch['intensity_future'].cpu().numpy()
        
        # Predict (returns denormalized values)
        results = predictor.predict_trajectory(
            past_frames,
            past_track,
            past_intensity,
            num_future_steps=target_track.shape[1],
            num_samples=num_samples
        )
        
        # Denormalize targets if they're normalized (for error computation)
        target_track_denorm = target_track.copy()
        target_intensity_denorm = target_intensity.copy()
        if np.abs(target_track).max() < 5:  # Likely normalized
            target_track_denorm[:, :, 0] = target_track[:, :, 0] * 12.5 + 22.5  # Latitude
            target_track_denorm[:, :, 1] = target_track[:, :, 1] * 20.0 + 140.0  # Longitude
            target_intensity_denorm = target_intensity * 26.5 + 43.5  # Wind speed
        
        # Compute errors (both in physical units now)
        track_error = np.linalg.norm(
            results['track_mean'] - target_track_denorm[0], axis=-1
        )  # (T,) - in degrees
        
        intensity_error = np.abs(
            results['intensity_mean'] - target_intensity_denorm[0]
        )  # (T,) - in m/s
        
        all_track_errors.append(track_error)
        all_intensity_errors.append(intensity_error)
        
        # Save first few predictions for visualization
        if save_dir and batch_idx < 10:
            save_prediction_plot(
                batch, results, batch_idx, save_dir
            )
    
    # Aggregate statistics
    all_track_errors = np.array(all_track_errors)  # (N_samples, T)
    all_intensity_errors = np.array(all_intensity_errors)  # (N_samples, T)
    
    # Compute metrics
    metrics = {
        'track_mae': all_track_errors.mean(),
        'track_mae_per_step': all_track_errors.mean(axis=0),
        'intensity_mae': all_intensity_errors.mean(),
        'intensity_mae_per_step': all_intensity_errors.mean(axis=0),
    }
    
    # Log results
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Track MAE: {metrics['track_mae']:.4f} degrees")
    logger.info(f"Intensity MAE: {metrics['intensity_mae']:.4f} m/s")
    logger.info("")
    logger.info("Per-step errors:")
    for t, (track_err, int_err) in enumerate(zip(
        metrics['track_mae_per_step'],
        metrics['intensity_mae_per_step']
    )):
        logger.info(f"  Step {t+1}: Track={track_err:.4f}°, Intensity={int_err:.4f} m/s")
    
    # Save metrics
    if save_dir:
        np.savez(
            save_dir / 'metrics.npz',
            **metrics
        )
        logger.info(f"Metrics saved to {save_dir / 'metrics.npz'}")
    
    return metrics


def save_prediction_plot(batch, results, idx, save_dir):
    """Save visualization of prediction"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Past trajectory (may be normalized, need to denormalize)
    past_track = batch['track_past'][0].cpu().numpy()
    target_track = batch['track_future'][0].cpu().numpy()
    pred_track = results['track_mean']  # Already denormalized in predictor
    track_std = results['track_std']
    
    # Denormalize past and target if they're normalized
    # Check if values are in normalized range (typically [-2, 2] for standardized data)
    if np.abs(past_track).max() < 5:  # Likely normalized
        past_track[:, 0] = past_track[:, 0] * 12.5 + 22.5  # Latitude
        past_track[:, 1] = past_track[:, 1] * 20.0 + 140.0  # Longitude
        target_track[:, 0] = target_track[:, 0] * 12.5 + 22.5  # Latitude
        target_track[:, 1] = target_track[:, 1] * 20.0 + 140.0  # Longitude
    
    # Plot trajectory
    ax = axes[0]
    ax.plot(past_track[:, 1], past_track[:, 0], 'b-o', label='Past', linewidth=2)
    ax.plot(target_track[:, 1], target_track[:, 0], 'g-o', label='Ground Truth', linewidth=2)
    ax.plot(pred_track[:, 1], pred_track[:, 0], 'r-o', label='Prediction', linewidth=2)
    
    # Uncertainty ellipses
    for t in range(len(pred_track)):
        circle = plt.Circle(
            (pred_track[t, 1], pred_track[t, 0]),
            track_std[t].mean(),
            color='r', alpha=0.2
        )
        ax.add_patch(circle)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajectory Prediction')
    ax.legend()
    ax.grid(True)
    
    # Plot intensity
    ax = axes[1]
    past_intensity = batch['intensity_past'][0].cpu().numpy()
    target_intensity = batch['intensity_future'][0].cpu().numpy()
    pred_intensity = results['intensity_mean']  # Already denormalized in predictor
    intensity_std = results['intensity_std']
    
    # Denormalize past and target if they're normalized
    if np.abs(past_intensity).max() < 5:  # Likely normalized
        past_intensity = past_intensity * 26.5 + 43.5  # Wind speed
        target_intensity = target_intensity * 26.5 + 43.5  # Wind speed
    
    time_past = np.arange(len(past_intensity))
    time_future = np.arange(len(past_intensity), len(past_intensity) + len(target_intensity))
    
    ax.plot(time_past, past_intensity, 'b-o', label='Past', linewidth=2)
    ax.plot(time_future, target_intensity, 'g-o', label='Ground Truth', linewidth=2)
    ax.plot(time_future, pred_intensity, 'r-o', label='Prediction', linewidth=2)
    ax.fill_between(
        time_future,
        pred_intensity - intensity_std,
        pred_intensity + intensity_std,
        color='r', alpha=0.2, label='Uncertainty'
    )
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Intensity Prediction')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'prediction_{idx:03d}.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Joint Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                       help='Path to joint autoencoder checkpoint')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                       help='Path to diffusion model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of ensemble samples')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (recommend 1 for evaluation)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictor
    logger.info("Loading models...")
    predictor = JointTrajectoryPredictor.from_checkpoints(
        autoencoder_path=args.autoencoder_checkpoint,
        diffusion_path=args.diffusion_checkpoint,
        config=config['model'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create test dataloader
    logger.info(f"Loading test data from {args.test_data}...")
    test_dataset = TyphoonDataset(
        data_dir=args.test_data,
        split='test',
        concat_ibtracs=False  # Joint model expects separate inputs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    # Evaluate
    metrics = evaluate_model(
        predictor,
        test_loader,
        num_samples=args.num_samples,
        save_dir=output_dir
    )
    
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

