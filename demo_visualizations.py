"""
Quick demonstration of visualization capabilities
Creates sample visualizations without running full training
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from visualize_results import TyphoonVisualizer


def create_sample_data():
    """Create sample data for demonstration"""
    
    # Sample typhoon track
    # Simulate a typhoon moving northwest
    t_past = 12
    t_future = 8
    
    # Past track (moving from southeast to northwest)
    past_lons = np.linspace(130, 135, t_past)
    past_lats = np.linspace(15, 20, t_past)
    past_track = np.stack([past_lons, past_lats], axis=1)
    
    # True future track (continuing northwest)
    true_lons = np.linspace(135, 138, t_future)
    true_lats = np.linspace(20, 24, t_future)
    true_track = np.stack([true_lons, true_lats], axis=1)
    
    # Predicted track (slightly off)
    pred_lons = true_lons + np.random.randn(t_future) * 0.5
    pred_lats = true_lats + np.random.randn(t_future) * 0.3
    pred_track = np.stack([pred_lons, pred_lats], axis=1)
    
    # Intensity (building up then weakening)
    past_intensity = np.array([20, 25, 30, 35, 40, 42, 45, 48, 50, 52, 53, 55])
    true_intensity = np.array([55, 56, 54, 50, 45, 40, 35, 30])
    pred_intensity = true_intensity + np.random.randn(t_future) * 2
    
    # Satellite-like frames
    def create_frames(track, intensity, t):
        frames = np.zeros((t, 4, 64, 64))  # 4 channels for display
        for i in range(t):
            y, x = 32, 32  # Center
            
            # Create intensity pattern
            yy, xx = np.ogrid[:64, :64]
            dist = np.sqrt((yy - y)**2 + (xx - x)**2)
            
            for c in range(4):
                pattern = (intensity[i] / 70.0) * np.exp(-dist / (8 + c * 2))
                noise = np.random.randn(64, 64) * 0.05
                frames[i, c] = pattern + noise
        
        return frames
    
    past_frames = create_frames(past_track, past_intensity, t_past)
    true_frames = create_frames(true_track, true_intensity, t_future)
    pred_frames = create_frames(pred_track, pred_intensity, t_future)
    
    return {
        'past_track': past_track,
        'true_track': true_track,
        'pred_track': pred_track,
        'past_intensity': past_intensity,
        'true_intensity': true_intensity,
        'pred_intensity': pred_intensity,
        'past_frames': past_frames,
        'true_frames': true_frames,
        'pred_frames': pred_frames
    }


def main():
    print("="*80)
    print("TYPHOON PREDICTION VISUALIZATION DEMO")
    print("="*80)
    print("\nGenerating sample data...")
    
    # Create output directory
    Path("demo_visualizations").mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = TyphoonVisualizer(output_dir="demo_visualizations")
    
    # Create multiple samples
    n_samples = 5
    all_track_errors = []
    all_intensity_errors = []
    
    print(f"\nCreating visualizations for {n_samples} samples...\n")
    
    for i in range(n_samples):
        print(f"Sample {i+1}/{n_samples}:")
        data = create_sample_data()
        
        # Trajectory
        track_error = visualizer.plot_trajectory_comparison(
            data['past_track'],
            data['true_track'],
            data['pred_track'],
            save_name=f"trajectory_sample_{i+1}.png"
        )
        all_track_errors.append(track_error)
        print(f"  ✓ Track error: {track_error:.1f} km")
        
        # Intensity
        intensity_metrics = visualizer.plot_intensity_comparison(
            data['past_intensity'],
            data['true_intensity'],
            data['pred_intensity'],
            save_name=f"intensity_sample_{i+1}.png"
        )
        all_intensity_errors.append(intensity_metrics['rmse'])
        print(f"  ✓ Intensity RMSE: {intensity_metrics['rmse']:.2f} m/s")
        
        # Frames
        visualizer.plot_frames_comparison(
            data['past_frames'],
            data['true_frames'],
            data['pred_frames'],
            channel_names=['Ch1', 'Ch2', 'Ch3', 'Ch4'],
            save_name=f"frames_sample_{i+1}.png"
        )
        print(f"  ✓ Frame comparison saved")
        
        # Animation
        combined_frames = np.concatenate([data['past_frames'], data['pred_frames']], axis=0)
        combined_track = np.concatenate([data['past_track'], data['pred_track']], axis=0)
        combined_intensity = np.concatenate([data['past_intensity'], data['pred_intensity']], axis=0)
        
        visualizer.create_animation(
            combined_frames,
            combined_track,
            combined_intensity,
            save_name=f"animation_sample_{i+1}.gif",
            fps=2
        )
        print(f"  ✓ Animation created")
    
    # Training curves (example)
    print("\nCreating training curves...")
    train_losses = [5000, 3500, 2500, 2000, 1800, 1600, 1500, 1400, 1350, 1300]
    val_losses = [5200, 3600, 2700, 2200, 1950, 1750, 1650, 1600, 1580, 1550]
    
    visualizer.plot_training_curves(
        train_losses,
        val_losses,
        model_name="Autoencoder",
        save_name="autoencoder_training_curves.png"
    )
    print("  ✓ Autoencoder training curves saved")
    
    visualizer.plot_training_curves(
        [loss * 1.5 for loss in train_losses],
        [loss * 1.5 for loss in val_losses],
        model_name="Diffusion Model",
        save_name="diffusion_training_curves.png"
    )
    print("  ✓ Diffusion training curves saved")
    
    # Error statistics
    print("\nCreating error statistics...")
    visualizer.plot_error_statistics(
        all_track_errors,
        all_intensity_errors
    )
    print("  ✓ Error statistics saved")
    
    # Create report
    print("\nGenerating HTML report...")
    results = {
        'timestamp': '2025-11-06',
        'mean_track_error': float(np.mean(all_track_errors)),
        'mean_intensity_rmse': float(np.mean(all_intensity_errors)),
        'n_samples': len(all_track_errors)
    }
    
    visualizer.create_comprehensive_report(results)
    print("  ✓ HTML report generated")
    
    # Summary
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE")
    print("="*80)
    print(f"\nResults saved to: demo_visualizations/")
    print(f"\nGenerated files:")
    print(f"  - {n_samples} trajectory comparisons")
    print(f"  - {n_samples} intensity comparisons")
    print(f"  - {n_samples} frame comparisons")
    print(f"  - {n_samples} animations (GIF)")
    print(f"  - 2 training curve plots")
    print(f"  - 1 error statistics plot")
    print(f"  - 1 HTML report")
    print(f"\nOpen demo_visualizations/prediction_report.html to view all results!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

