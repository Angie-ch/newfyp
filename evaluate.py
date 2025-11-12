"""
Evaluation Script for Typhoon Predictions

Computes metrics and generates visualizations
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TyphoonEvaluator:
    """
    Evaluate typhoon prediction performance
    """
    
    def __init__(self, predictions_file: str, output_dir: str):
        """
        Initialize evaluator
        
        Args:
            predictions_file: Path to predictions .npz file
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load predictions
        logger.info(f"Loading predictions from {predictions_file}")
        data = np.load(predictions_file, allow_pickle=True)
        self.predictions = data['predictions']
        self.ground_truth = data['ground_truth']
        
        logger.info(f"Loaded {len(self.predictions)} predictions")
    
    def compute_track_error(self) -> dict:
        """
        Compute track prediction error in kilometers
        
        Returns:
            Dictionary with error statistics
        """
        logger.info("Computing track errors...")
        
        errors = []
        
        for pred, gt in zip(self.predictions, self.ground_truth):
            if pred['track'] is not None and gt['track'] is not None:
                # Convert to numpy
                pred_track = pred['track'].numpy() if torch.is_tensor(pred['track']) else pred['track']
                gt_track = gt['track'].numpy() if torch.is_tensor(gt['track']) else gt['track']
                
                # Compute distance in km (approximate)
                # 1 degree latitude ≈ 111 km
                # 1 degree longitude ≈ 111 * cos(lat) km
                
                lat_diff = (pred_track[..., 0] - gt_track[..., 0]) * 111.0
                lon_diff = (pred_track[..., 1] - gt_track[..., 1]) * 111.0 * np.cos(np.radians(gt_track[..., 0]))
                
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                errors.append(distance)
        
        if not errors:
            logger.warning("No track predictions found")
            return {}
        
        errors = np.concatenate(errors, axis=0)
        
        results = {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'per_timestep': [np.mean(errors[:, t]) for t in range(errors.shape[1])]
        }
        
        logger.info(f"Track Error - Mean: {results['mean']:.2f} km, Median: {results['median']:.2f} km")
        
        return results
    
    def compute_intensity_error(self) -> dict:
        """
        Compute intensity prediction error (wind speed)
        
        Returns:
            Dictionary with error statistics
        """
        logger.info("Computing intensity errors...")
        
        errors = []
        
        for pred, gt in zip(self.predictions, self.ground_truth):
            if pred['intensity'] is not None and gt['intensity'] is not None:
                pred_intensity = pred['intensity'].numpy() if torch.is_tensor(pred['intensity']) else pred['intensity']
                gt_intensity = gt['intensity'].numpy() if torch.is_tensor(gt['intensity']) else gt['intensity']
                
                error = np.abs(pred_intensity - gt_intensity)
                errors.append(error)
        
        if not errors:
            logger.warning("No intensity predictions found")
            return {}
        
        errors = np.concatenate(errors, axis=0)
        
        results = {
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'median': np.median(errors),
            'per_timestep': [np.mean(errors[:, t]) for t in range(errors.shape[1])]
        }
        
        logger.info(f"Intensity Error - MAE: {results['mae']:.2f} m/s, RMSE: {results['rmse']:.2f} m/s")
        
        return results
    
    def compute_field_error(self) -> dict:
        """
        Compute atmospheric field reconstruction error
        
        Returns:
            Dictionary with error statistics
        """
        logger.info("Computing field reconstruction errors...")
        
        mse_errors = []
        mae_errors = []
        
        for pred, gt in zip(self.predictions, self.ground_truth):
            pred_frames = pred['future_frames']
            gt_frames = gt['future_frames']
            
            # Convert to numpy
            if torch.is_tensor(pred_frames):
                pred_frames = pred_frames.numpy()
            if torch.is_tensor(gt_frames):
                gt_frames = gt_frames.numpy()
            
            # Compute errors
            mse = np.mean((pred_frames - gt_frames)**2, axis=(1, 2, 3, 4))  # Average over all dims except batch
            mae = np.mean(np.abs(pred_frames - gt_frames), axis=(1, 2, 3, 4))
            
            mse_errors.append(mse)
            mae_errors.append(mae)
        
        mse_errors = np.concatenate(mse_errors, axis=0)
        mae_errors = np.concatenate(mae_errors, axis=0)
        
        results = {
            'mse': np.mean(mse_errors),
            'rmse': np.sqrt(np.mean(mse_errors)),
            'mae': np.mean(mae_errors),
            'per_timestep_mse': [np.mean(mse_errors[:, t]) for t in range(mse_errors.shape[1])],
            'per_timestep_mae': [np.mean(mae_errors[:, t]) for t in range(mae_errors.shape[1])]
        }
        
        logger.info(f"Field Error - MSE: {results['mse']:.4f}, MAE: {results['mae']:.4f}")
        
        return results
    
    def plot_track_comparison(self, sample_idx: int = 0):
        """Plot predicted vs actual track"""
        logger.info(f"Plotting track comparison for sample {sample_idx}...")
        
        pred = self.predictions[sample_idx]
        gt = self.ground_truth[sample_idx]
        
        if pred['track'] is None or gt['track'] is None:
            logger.warning("No track data available")
            return
        
        pred_track = pred['track'].numpy() if torch.is_tensor(pred['track']) else pred['track']
        gt_track = gt['track'].numpy() if torch.is_tensor(gt['track']) else gt['track']
        
        # Handle batch dimension
        if pred_track.ndim == 3:
            pred_track = pred_track[0]
            gt_track = gt_track[0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot tracks
        ax.plot(gt_track[:, 1], gt_track[:, 0], 'o-', 
               label='Ground Truth', color='blue', linewidth=2, markersize=8)
        ax.plot(pred_track[:, 1], pred_track[:, 0], 's--', 
               label='Predicted', color='red', linewidth=2, markersize=8)
        
        # Mark start and end
        ax.plot(gt_track[0, 1], gt_track[0, 0], '*', 
               color='green', markersize=20, label='Start')
        ax.plot(gt_track[-1, 1], gt_track[-1, 0], 'X', 
               color='orange', markersize=15, label='End')
        
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_title(f'Typhoon Track Prediction - Sample {sample_idx}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'track_comparison_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved track plot to {self.output_dir / f'track_comparison_{sample_idx}.png'}")
    
    def plot_intensity_comparison(self, sample_idx: int = 0):
        """Plot predicted vs actual intensity"""
        logger.info(f"Plotting intensity comparison for sample {sample_idx}...")
        
        pred = self.predictions[sample_idx]
        gt = self.ground_truth[sample_idx]
        
        if pred['intensity'] is None or gt['intensity'] is None:
            logger.warning("No intensity data available")
            return
        
        pred_intensity = pred['intensity'].numpy() if torch.is_tensor(pred['intensity']) else pred['intensity']
        gt_intensity = gt['intensity'].numpy() if torch.is_tensor(gt['intensity']) else gt['intensity']
        
        # Handle batch dimension
        if pred_intensity.ndim == 2:
            pred_intensity = pred_intensity[0]
            gt_intensity = gt_intensity[0]
        
        timesteps = np.arange(len(gt_intensity)) * 6  # 6-hour intervals
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(timesteps, gt_intensity, 'o-', 
               label='Ground Truth', color='blue', linewidth=2, markersize=8)
        ax.plot(timesteps, pred_intensity, 's--', 
               label='Predicted', color='red', linewidth=2, markersize=8)
        
        ax.set_xlabel('Forecast Hour', fontsize=12)
        ax.set_ylabel('Maximum Wind Speed (m/s)', fontsize=12)
        ax.set_title(f'Intensity Prediction - Sample {sample_idx}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'intensity_comparison_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved intensity plot to {self.output_dir / f'intensity_comparison_{sample_idx}.png'}")
    
    def plot_error_evolution(self, track_errors: dict, intensity_errors: dict):
        """Plot how errors evolve with forecast time"""
        logger.info("Plotting error evolution...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Track error
        if 'per_timestep' in track_errors:
            timesteps = np.arange(len(track_errors['per_timestep'])) * 6
            ax1.plot(timesteps, track_errors['per_timestep'], 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Forecast Hour', fontsize=12)
            ax1.set_ylabel('Track Error (km)', fontsize=12)
            ax1.set_title('Track Error vs Forecast Time', fontsize=14)
            ax1.grid(True, alpha=0.3)
        
        # Intensity error
        if 'per_timestep' in intensity_errors:
            timesteps = np.arange(len(intensity_errors['per_timestep'])) * 6
            ax2.plot(timesteps, intensity_errors['per_timestep'], 'o-', 
                    color='orange', linewidth=2, markersize=8)
            ax2.set_xlabel('Forecast Hour', fontsize=12)
            ax2.set_ylabel('Intensity Error (m/s)', fontsize=12)
            ax2.set_title('Intensity Error vs Forecast Time', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error evolution plot to {self.output_dir / 'error_evolution.png'}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Compute metrics
        track_errors = self.compute_track_error()
        intensity_errors = self.compute_intensity_error()
        field_errors = self.compute_field_error()
        
        # Generate plots
        if len(self.predictions) > 0:
            self.plot_track_comparison(0)
            self.plot_intensity_comparison(0)
            
            if len(self.predictions) > 1:
                self.plot_track_comparison(1)
                self.plot_intensity_comparison(1)
        
        if track_errors and intensity_errors:
            self.plot_error_evolution(track_errors, intensity_errors)
        
        # Save metrics to file
        report_file = self.output_dir / 'evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Typhoon Prediction Evaluation Report\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Number of samples: {len(self.predictions)}\n\n")
            
            if track_errors:
                f.write("Track Prediction Errors (km):\n")
                f.write(f"  Mean:   {track_errors['mean']:.2f}\n")
                f.write(f"  Median: {track_errors['median']:.2f}\n")
                f.write(f"  Std:    {track_errors['std']:.2f}\n")
                f.write(f"  Min:    {track_errors['min']:.2f}\n")
                f.write(f"  Max:    {track_errors['max']:.2f}\n\n")
            
            if intensity_errors:
                f.write("Intensity Prediction Errors (m/s):\n")
                f.write(f"  MAE:    {intensity_errors['mae']:.2f}\n")
                f.write(f"  RMSE:   {intensity_errors['rmse']:.2f}\n")
                f.write(f"  Median: {intensity_errors['median']:.2f}\n\n")
            
            if field_errors:
                f.write("Atmospheric Field Errors:\n")
                f.write(f"  MSE:    {field_errors['mse']:.6f}\n")
                f.write(f"  RMSE:   {field_errors['rmse']:.6f}\n")
                f.write(f"  MAE:    {field_errors['mae']:.6f}\n\n")
        
        logger.info(f"Saved evaluation report to {report_file}")
        logger.info("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Typhoon Predictions')
    parser.add_argument('--predictions', type=str, required=True,
                      help='Path to predictions .npz file')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                      help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = TyphoonEvaluator(
        predictions_file=args.predictions,
        output_dir=args.output_dir
    )
    
    # Generate report
    evaluator.generate_report()


if __name__ == '__main__':
    main()
