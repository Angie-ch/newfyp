"""
Visualization utilities for typhoon prediction results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns

sns.set_style("whitegrid")


class TyphoonVisualizer:
    """Comprehensive visualization for typhoon predictions"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_curves(self, 
                            train_losses: List[float], 
                            val_losses: List[float],
                            model_name: str = "model",
                            save_name: Optional[str] = None):
        """Plot training and validation loss curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{model_name} Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"{model_name.lower().replace(' ', '_')}_training_curves.png"
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
        plt.close()
        
    def plot_trajectory_comparison(self,
                                   past_track: np.ndarray,
                                   true_track: np.ndarray,
                                   pred_track: np.ndarray,
                                   save_name: str = "trajectory_comparison.png"):
        """
        Plot past trajectory and compare predicted vs true future trajectories
        
        Args:
            past_track: (T_past, 2) array of (lon, lat)
            true_track: (T_future, 2) array of (lon, lat) 
            pred_track: (T_future, 2) array of (lon, lat)
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot past trajectory
        ax.plot(past_track[:, 0], past_track[:, 1], 'ko-', 
                linewidth=3, markersize=8, label='Past Track', zorder=3)
        
        # Plot true future trajectory
        ax.plot(true_track[:, 0], true_track[:, 1], 'g^-', 
                linewidth=3, markersize=10, label='True Future', zorder=2)
        
        # Plot predicted trajectory
        ax.plot(pred_track[:, 0], pred_track[:, 1], 'r*--', 
                linewidth=3, markersize=12, label='Predicted', zorder=2, alpha=0.8)
        
        # Mark start and end points
        ax.plot(past_track[0, 0], past_track[0, 1], 'bs', 
                markersize=15, label='Start', zorder=4)
        ax.plot(true_track[-1, 0], true_track[-1, 1], 'gs', 
                markersize=15, label='True End', zorder=4)
        ax.plot(pred_track[-1, 0], pred_track[-1, 1], 'rs', 
                markersize=15, label='Pred End', zorder=4)
        
        # Connect past to future
        ax.plot([past_track[-1, 0], true_track[0, 0]], 
                [past_track[-1, 1], true_track[0, 1]], 
                'g-', linewidth=2, alpha=0.5)
        ax.plot([past_track[-1, 0], pred_track[0, 0]], 
                [past_track[-1, 1], pred_track[0, 1]], 
                'r--', linewidth=2, alpha=0.5)
        
        # Calculate error
        error_km = np.sqrt(np.sum((true_track - pred_track)**2, axis=1)) * 111  # approx km/degree
        mean_error = np.mean(error_km)
        
        ax.set_xlabel('Longitude (°E)', fontsize=13)
        ax.set_ylabel('Latitude (°N)', fontsize=13)
        ax.set_title(f'Typhoon Track Prediction\nMean Error: {mean_error:.1f} km', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved trajectory comparison to {save_path}")
        plt.close()
        
        return mean_error
        
    def plot_intensity_comparison(self,
                                  past_intensity: np.ndarray,
                                  true_intensity: np.ndarray,
                                  pred_intensity: np.ndarray,
                                  time_hours: Optional[np.ndarray] = None,
                                  save_name: str = "intensity_comparison.png"):
        """
        Plot intensity evolution comparison
        
        Args:
            past_intensity: (T_past,) array of past intensities
            true_intensity: (T_future,) array of true future intensities
            pred_intensity: (T_future,) array of predicted intensities
            time_hours: optional time array in hours
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        T_past = len(past_intensity)
        T_future = len(true_intensity)
        
        if time_hours is None:
            time_hours = np.arange(T_past + T_future) * 6  # 6-hour intervals
        else:
            time_past = time_hours[:T_past]
            time_future = time_hours[T_past:]
        
        time_past = time_hours[:T_past]
        time_future = time_hours[T_past:T_past+T_future]
        
        # Plot past
        ax.plot(time_past, past_intensity, 'ko-', 
                linewidth=3, markersize=8, label='Past Intensity')
        
        # Plot true future
        ax.plot(time_future, true_intensity, 'g^-', 
                linewidth=3, markersize=10, label='True Future')
        
        # Plot predicted
        ax.plot(time_future, pred_intensity, 'r*--', 
                linewidth=3, markersize=12, label='Predicted', alpha=0.8)
        
        # Add vertical line at forecast start
        ax.axvline(x=time_past[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(time_past[-1], ax.get_ylim()[1]*0.95, 'Forecast Start', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((true_intensity - pred_intensity)**2))
        mae = np.mean(np.abs(true_intensity - pred_intensity))
        
        ax.set_xlabel('Time (hours)', fontsize=13)
        ax.set_ylabel('Intensity (m/s)', fontsize=13)
        ax.set_title(f'Typhoon Intensity Prediction\nRMSE: {rmse:.2f} m/s, MAE: {mae:.2f} m/s', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved intensity comparison to {save_path}")
        plt.close()
        
        return {'rmse': rmse, 'mae': mae}
        
    def plot_frames_comparison(self,
                              past_frames: np.ndarray,
                              true_frames: np.ndarray,
                              pred_frames: np.ndarray,
                              channel_names: List[str] = None,
                              save_name: str = "frames_comparison.png"):
        """
        Plot side-by-side comparison of satellite frames
        
        Args:
            past_frames: (T_past, C, H, W) array
            true_frames: (T_future, C, H, W) array
            pred_frames: (T_future, C, H, W) array
            channel_names: list of channel names to display
        """
        T_past, C, H, W = past_frames.shape
        T_future = true_frames.shape[0]
        
        # Select channels to visualize (e.g., first few channels)
        if channel_names is None:
            channel_names = [f'Ch{i}' for i in range(min(C, 4))]
        
        n_channels = min(len(channel_names), 4)
        
        # Select time steps to show
        show_past_steps = min(3, T_past)
        show_future_steps = min(4, T_future)
        
        past_indices = np.linspace(0, T_past-1, show_past_steps, dtype=int)
        future_indices = np.linspace(0, T_future-1, show_future_steps, dtype=int)
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(n_channels, show_past_steps + 2*show_future_steps, 
                     figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot past frames
        for ch_idx in range(n_channels):
            for col_idx, t_idx in enumerate(past_indices):
                ax = fig.add_subplot(gs[ch_idx, col_idx])
                im = ax.imshow(past_frames[t_idx, ch_idx], cmap='viridis')
                
                if ch_idx == 0:
                    ax.set_title(f'Past t={t_idx}', fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(channel_names[ch_idx], fontsize=11, fontweight='bold')
                ax.axis('off')
                
        # Plot true future frames
        for ch_idx in range(n_channels):
            for col_idx, t_idx in enumerate(future_indices):
                ax = fig.add_subplot(gs[ch_idx, show_past_steps + col_idx])
                im = ax.imshow(true_frames[t_idx, ch_idx], cmap='viridis')
                
                if ch_idx == 0:
                    ax.set_title(f'True t={t_idx}', fontsize=10, fontweight='bold', color='green')
                ax.axis('off')
                
        # Plot predicted future frames
        for ch_idx in range(n_channels):
            for col_idx, t_idx in enumerate(future_indices):
                ax = fig.add_subplot(gs[ch_idx, show_past_steps + show_future_steps + col_idx])
                im = ax.imshow(pred_frames[t_idx, ch_idx], cmap='viridis')
                
                if ch_idx == 0:
                    ax.set_title(f'Pred t={t_idx}', fontsize=10, fontweight='bold', color='red')
                ax.axis('off')
        
        plt.suptitle('Satellite Imagery: Past → True Future → Predicted Future', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved frames comparison to {save_path}")
        plt.close()
        
    def create_animation(self,
                        frames: np.ndarray,
                        track: Optional[np.ndarray] = None,
                        intensity: Optional[np.ndarray] = None,
                        save_name: str = "typhoon_animation.gif",
                        fps: int = 2):
        """
        Create animation of typhoon evolution
        
        Args:
            frames: (T, C, H, W) array - uses first channel for visualization
            track: Optional (T, 2) array of (lon, lat)
            intensity: Optional (T,) array of intensities
        """
        T, C, H, W = frames.shape
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initialize image
        im = axes[0].imshow(frames[0, 0], cmap='viridis', animated=True)
        axes[0].set_title('Satellite Imagery', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Initialize track plot if provided
        if track is not None:
            line, = axes[1].plot([], [], 'ro-', linewidth=2, markersize=8)
            current_point, = axes[1].plot([], [], 'b*', markersize=20)
            axes[1].set_xlim(track[:, 0].min() - 1, track[:, 0].max() + 1)
            axes[1].set_ylim(track[:, 1].min() - 1, track[:, 1].max() + 1)
            axes[1].set_xlabel('Longitude', fontsize=11)
            axes[1].set_ylabel('Latitude', fontsize=11)
            axes[1].set_title('Track', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_aspect('equal')
        
        time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12, fontweight='bold')
        
        def animate(i):
            im.set_array(frames[i, 0])
            
            if track is not None:
                line.set_data(track[:i+1, 0], track[:i+1, 1])
                current_point.set_data([track[i, 0]], [track[i, 1]])
            
            time_info = f'Time Step: {i+1}/{T}'
            if intensity is not None:
                time_info += f', Intensity: {intensity[i]:.1f} m/s'
            time_text.set_text(time_info)
            
            return [im, line, current_point, time_text] if track is not None else [im, time_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=T, 
                                      interval=1000//fps, blit=True)
        
        save_path = self.output_dir / save_name
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"✓ Saved animation to {save_path}")
        plt.close()
        
    def plot_error_statistics(self,
                             track_errors: List[float],
                             intensity_errors: List[float],
                             save_name: str = "error_statistics.png"):
        """
        Plot error statistics across multiple predictions
        
        Args:
            track_errors: List of mean track errors (km) for each sample
            intensity_errors: List of RMSE intensity errors for each sample
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Track error histogram
        axes[0].hist(track_errors, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(track_errors), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(track_errors):.1f} km')
        axes[0].axvline(np.median(track_errors), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(track_errors):.1f} km')
        axes[0].set_xlabel('Track Error (km)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Track Error Distribution', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Intensity error histogram
        axes[1].hist(intensity_errors, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(intensity_errors), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(intensity_errors):.2f} m/s')
        axes[1].axvline(np.median(intensity_errors), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(intensity_errors):.2f} m/s')
        axes[1].set_xlabel('Intensity RMSE (m/s)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Intensity Error Distribution', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved error statistics to {save_path}")
        plt.close()
        
    def create_comprehensive_report(self,
                                   results: Dict,
                                   save_name: str = "prediction_report.html"):
        """
        Create HTML report summarizing all predictions
        
        Args:
            results: Dictionary containing all prediction results and metrics
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Typhoon Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: white; padding: 20px; margin: 20px 0; 
                          border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                        gap: 20px; }}
                img {{ max-width: 100%; height: auto; border-radius: 8px; 
                      box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; background-color: white;
                        margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>🌀 Typhoon Prediction Report</h1>
            <p><strong>Generated:</strong> {results.get('timestamp', 'N/A')}</p>
            
            <h2>📊 Summary Metrics</h2>
            <div class="grid">
                <div class="metric">
                    <div class="metric-label">Mean Track Error</div>
                    <div class="metric-value">{results.get('mean_track_error', 0):.1f} km</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Intensity RMSE</div>
                    <div class="metric-value">{results.get('mean_intensity_rmse', 0):.2f} m/s</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Samples Evaluated</div>
                    <div class="metric-value">{results.get('n_samples', 0)}</div>
                </div>
            </div>
            
            <h2>📈 Training Progress</h2>
            <img src="training_curves.png" alt="Training Curves">
            
            <h2>🗺️ Trajectory Predictions</h2>
            <img src="trajectory_comparison.png" alt="Trajectory Comparison">
            
            <h2>💨 Intensity Predictions</h2>
            <img src="intensity_comparison.png" alt="Intensity Comparison">
            
            <h2>🛰️ Satellite Imagery</h2>
            <img src="frames_comparison.png" alt="Frames Comparison">
            
            <h2>📉 Error Analysis</h2>
            <img src="error_statistics.png" alt="Error Statistics">
            
        </body>
        </html>
        """
        
        save_path = self.output_dir / save_name
        with open(save_path, 'w') as f:
            f.write(html)
        print(f"✓ Saved comprehensive report to {save_path}")


def visualize_from_predictions(predictions_file: str, output_dir: str = "visualizations"):
    """
    Load predictions from file and create all visualizations
    
    Args:
        predictions_file: Path to saved predictions (.npz or .pt file)
        output_dir: Directory to save visualizations
    """
    visualizer = TyphoonVisualizer(output_dir)
    
    # Load predictions
    if predictions_file.endswith('.npz'):
        data = np.load(predictions_file)
    elif predictions_file.endswith('.pt'):
        data = torch.load(predictions_file, map_location='cpu')
        # Convert to numpy
        data = {k: v.numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in data.items()}
    else:
        raise ValueError("Predictions file must be .npz or .pt")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Extract data
    past_track = data.get('past_track')
    true_track = data.get('true_track')
    pred_track = data.get('pred_track')
    past_intensity = data.get('past_intensity')
    true_intensity = data.get('true_intensity')
    pred_intensity = data.get('pred_intensity')
    past_frames = data.get('past_frames')
    true_frames = data.get('true_frames')
    pred_frames = data.get('pred_frames')
    
    # Trajectory
    if past_track is not None and true_track is not None and pred_track is not None:
        visualizer.plot_trajectory_comparison(past_track, true_track, pred_track)
    
    # Intensity
    if past_intensity is not None and true_intensity is not None and pred_intensity is not None:
        visualizer.plot_intensity_comparison(past_intensity, true_intensity, pred_intensity)
    
    # Frames
    if past_frames is not None and true_frames is not None and pred_frames is not None:
        visualizer.plot_frames_comparison(past_frames, true_frames, pred_frames)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
        visualize_from_predictions(predictions_file, output_dir)
    else:
        print("Usage: python visualize_results.py <predictions_file> [output_dir]")

