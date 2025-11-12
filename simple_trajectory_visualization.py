"""
Simple Trajectory Visualization Script

Works with existing preprocessed data to demonstrate the visualization concept.
This is a simplified version that doesn't require trained models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import logging

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available, using simple matplotlib plots")

from data.datasets.typhoon_dataset import TyphoonDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_trajectory_simple(
    past_track: np.ndarray,
    future_track: np.ndarray,
    case_id: str,
    save_path: str,
    past_intensity: np.ndarray = None,
    future_intensity: np.ndarray = None,
    satellite_bg: str = None
):
    """
    Visualize typhoon trajectory with past and future components
    
    Args:
        past_track: (T_past, 2) array of [lat, lon] for past timesteps
        future_track: (T_future, 2) array of [lat, lon] for future timesteps
        case_id: Case identifier
        save_path: Where to save the figure
        past_intensity: Optional (T_past,) intensity values
        future_intensity: Optional (T_future,) intensity values
        satellite_bg: Optional path to satellite background image
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Left panel: Trajectory map
    if HAS_CARTOPY:
        ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        
        # Set extent based on track data
        all_lons = np.concatenate([past_track[:, 1], future_track[:, 1]])
        all_lats = np.concatenate([past_track[:, 0], future_track[:, 0]])
        
        lon_min, lon_max = all_lons.min() - 5, all_lons.max() + 5
        lat_min, lat_max = all_lats.min() - 5, all_lats.max() + 5
        
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax1.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
        ax1.gridlines(draw_labels=True, alpha=0.3, linewidth=0.5)
        
        transform = ccrs.PlateCarree()
    else:
        ax1 = plt.subplot(1, 2, 1)
        transform = None
    
    # Add satellite background if provided
    if satellite_bg and Path(satellite_bg).exists():
        try:
            img = Image.open(satellite_bg)
            lon_min, lon_max = past_track[:, 1].min() - 10, past_track[:, 1].max() + 10
            lat_min, lat_max = past_track[:, 0].min() - 10, past_track[:, 0].max() + 10
            extent = [lon_min, lon_max, lat_min, lat_max]
            ax1.imshow(img, extent=extent, transform=transform, alpha=0.4, zorder=0)
        except Exception as e:
            logger.warning(f"Could not load satellite background: {e}")
    
    # Plot PAST trajectory (input - blue/black circles)
    ax1.plot(past_track[:, 1], past_track[:, 0], 'ko-', 
             linewidth=3, markersize=10, label=f'Past ({len(past_track)} steps)', 
             zorder=3, transform=transform)
    
    # Mark starting point (blue square)
    ax1.plot(past_track[0, 1], past_track[0, 0], 'bs', 
             markersize=15, label='Start', zorder=4, transform=transform)
    
    # Plot FUTURE trajectory (ground truth - green triangles)
    ax1.plot(future_track[:, 1], future_track[:, 0], 'g^-', 
             linewidth=3, markersize=12, label=f'Future ({len(future_track)} steps)', 
             zorder=2, transform=transform)
    
    # Mark ending point (green square)
    ax1.plot(future_track[-1, 1], future_track[-1, 0], 'gs', 
             markersize=15, label='End', zorder=4, transform=transform)
    
    # Connect past to future with dashed line
    ax1.plot([past_track[-1, 1], future_track[0, 1]], 
             [past_track[-1, 0], future_track[0, 0]], 
             'k--', linewidth=2, alpha=0.5, zorder=1, transform=transform)
    
    # Add time labels at key points
    # Start
    ax1.text(past_track[0, 1], past_track[0, 0] + 0.5, 't=0h', 
             fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
             transform=transform)
    
    # Forecast start
    ax1.text(past_track[-1, 1], past_track[-1, 0] + 0.5, f't={len(past_track)*6}h', 
             fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             transform=transform)
    
    # End
    total_hours = (len(past_track) + len(future_track)) * 6
    ax1.text(future_track[-1, 1], future_track[-1, 0] + 0.5, f't={total_hours}h', 
             fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
             transform=transform)
    
    ax1.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Typhoon Track: {case_id}\n'
                  f'Input: {len(past_track)} steps ({len(past_track)*6}h) → '
                  f'Forecast: {len(future_track)} steps ({len(future_track)*6}h)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Coordinate evolution over time
    ax2 = plt.subplot(1, 2, 2)
    
    # Time axis (hours)
    time_past = np.arange(len(past_track)) * 6
    time_future = np.arange(len(past_track), len(past_track) + len(future_track)) * 6
    
    # Plot latitude evolution
    ax2.plot(time_past, past_track[:, 0], 'ko-', linewidth=2.5, markersize=9, 
             label='Past Latitude')
    ax2.plot(time_future, future_track[:, 0], 'g^-', linewidth=2.5, markersize=11, 
             label='Future Latitude')
    
    # Plot longitude evolution
    ax2.plot(time_past, past_track[:, 1], 'ks-', linewidth=2.5, markersize=9, 
             label='Past Longitude')
    ax2.plot(time_future, future_track[:, 1], 'gv-', linewidth=2.5, markersize=11, 
             label='Future Longitude')
    
    # Add forecast start line
    ax2.axvline(x=time_past[-1], color='orange', linestyle=':', linewidth=3, alpha=0.8, zorder=1)
    ax2.text(time_past[-1], ax2.get_ylim()[1]*0.95, 'Forecast Start', 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.6))
    
    # If intensity data is available, plot on secondary axis
    if past_intensity is not None and future_intensity is not None:
        ax2_intensity = ax2.twinx()
        ax2_intensity.plot(time_past, past_intensity, 'b*--', linewidth=2, markersize=10,
                          label='Past Intensity', alpha=0.6)
        ax2_intensity.plot(time_future, future_intensity, 'r*--', linewidth=2, markersize=10,
                          label='Future Intensity', alpha=0.6)
        ax2_intensity.set_ylabel('Intensity (m/s)', fontsize=13, fontweight='bold', color='blue')
        ax2_intensity.tick_params(axis='y', labelcolor='blue')
        ax2_intensity.legend(fontsize=10, loc='upper right')
    
    ax2.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Coordinates (degrees)', fontsize=14, fontweight='bold')
    ax2.set_title('Coordinate Evolution Over Time', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Simple Trajectory Visualization')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='visualizations/trajectories',
                       help='Where to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--satellite_bg', type=str, default=None,
                       help='Optional: Path to satellite background image')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which dataset split to use')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading dataset...")
    dataset = TyphoonDataset(
        data_dir=args.data_dir,
        split=args.split,
        normalize=False,  # Keep in original coordinates for visualization
        concat_ibtracs=False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        logger.error("No data found! Please check your data directory.")
        return
    
    # Process samples
    num_to_visualize = min(args.num_samples, len(dataset))
    logger.info(f"\nGenerating {num_to_visualize} trajectory visualizations...")
    
    for idx in range(num_to_visualize):
        logger.info(f"\nProcessing sample {idx + 1}/{num_to_visualize}")
        
        try:
            sample = dataset[idx]
            
            # Extract trajectory data
            past_track = sample['track_past'].cpu().numpy()  # (T_past, 2) [lat, lon]
            future_track = sample['track_future'].cpu().numpy()  # (T_future, 2)
            
            case_id = sample.get('case_id', f'sample_{idx}')
            
            # Extract intensity if available
            past_intensity = sample.get('intensity_past')
            future_intensity = sample.get('intensity_future')
            
            if past_intensity is not None:
                past_intensity = past_intensity.cpu().numpy()
            if future_intensity is not None:
                future_intensity = future_intensity.cpu().numpy()
            
            logger.info(f"  Case ID: {case_id}")
            logger.info(f"  Past trajectory: {len(past_track)} points ({len(past_track)*6} hours)")
            logger.info(f"  Future trajectory: {len(future_track)} points ({len(future_track)*6} hours)")
            
            # Generate visualization
            save_path = output_dir / f'trajectory_{case_id}.png'
            
            visualize_trajectory_simple(
                past_track=past_track,
                future_track=future_track,
                case_id=case_id,
                save_path=str(save_path),
                past_intensity=past_intensity,
                future_intensity=future_intensity,
                satellite_bg=args.satellite_bg
            )
            
            logger.info(f"  ✓ Saved to {save_path}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Generated {num_to_visualize} visualizations")
    logger.info(f"✓ Saved to: {output_dir}/")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()















