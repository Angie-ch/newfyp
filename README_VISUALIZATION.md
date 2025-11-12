# Typhoon Prediction - Visualization and Real Data Guide

## Overview

This project now includes comprehensive visualization tools and real typhoon data loading capabilities.

## New Features Added

### 1. Visualization Module (`visualize_results.py`)

The `TyphoonVisualizer` class provides:

- **Training Curves**: Plot training and validation losses over epochs
- **Trajectory Comparison**: Side-by-side comparison of predicted vs actual typhoon tracks
- **Intensity Comparison**: Time series plots of predicted vs actual intensity
- **Satellite Imagery Frames**: Visual comparison of past, true future, and predicted satellite imagery
- **Animations**: Animated GIFs showing typhoon evolution with track overlay
- **Error Statistics**: Histograms and summary statistics of prediction errors
- **HTML Reports**: Comprehensive HTML reports combining all visualizations

#### Usage:

```python
from visualize_results import TyphoonVisualizer

visualizer = TyphoonVisualizer(output_dir="visualizations")

# Plot training curves
visualizer.plot_training_curves(train_losses, val_losses, model_name="Autoencoder")

# Compare trajectories
visualizer.plot_trajectory_comparison(past_track, true_track, pred_track)

# Compare intensities
visualizer.plot_intensity_comparison(past_intensity, true_intensity, pred_intensity)

# Visualize satellite frames
visualizer.plot_frames_comparison(past_frames, true_frames, pred_frames)

# Create animation
visualizer.create_animation(frames, track, intensity, save_name="typhoon.gif")

# Generate comprehensive report
visualizer.create_comprehensive_report(results_dict)
```

### 2. Real Data Loader (`data/real_data_loader.py`)

The `IBTrACSLoader` class loads real typhoon data from the IBTrACS database:

**Features:**
- Automatically downloads IBTrACS WP (Western Pacific) typhoon track data
- Integrates with ERA5 reanalysis for meteorological data (48 channels)
- Filters for strong typhoons based on customizable criteria
- Creates training samples with past/future splits
- Falls back to synthetic frames if ERA5 not available

**Data Sources:**
- **Track data**: IBTrACS Western Pacific (automatic download)
- **Meteorological data**: ERA5 reanalysis (requires CDS API setup - see `ERA5_SETUP.md`)
- **Fallback**: Synthetic frames (no setup required)

#### Usage:

```python
from data.real_data_loader import IBTrACSLoader

loader = IBTrACSLoader(data_dir="data/raw")

# Create dataset with ERA5 (if available)
samples = loader.create_dataset(
    n_samples=100,
    start_year=2018,
    end_year=2023,
    past_timesteps=12,
    future_timesteps=8,
    use_era5=True,          # Use ERA5 data if available
    download_era5=False,    # Set to True to download ERA5 data
    save_path="data/processed/typhoons_wp_era5.npz"
)

# Or without ERA5 (synthetic frames)
samples = loader.create_dataset(
    n_samples=100,
    use_era5=False,
    save_path="data/processed/typhoons_wp_synthetic.npz"
)
```

### 3. Complete Test Script (`test_with_real_data.py`)

Comprehensive end-to-end pipeline test that:

1. Loads real typhoon data from IBTrACS (or falls back to synthetic)
2. Creates PyTorch datasets
3. Trains autoencoder model
4. Trains diffusion model
5. Generates predictions on test set
6. Creates all visualizations
7. Generates HTML report

#### Running the test:

```bash
python test_with_real_data.py
```

**Output locations:**
- Visualizations: `visualizations/`
- Model checkpoints: `checkpoints/`
- HTML report: `visualizations/prediction_report.html`
- Training logs: `test_real_data.log`

## Visualizations Generated

### 1. Training Curves
- `autoencoder_training_curves.png`
- `diffusion_training_curves.png`

Shows loss evolution over training epochs for both models.

### 2. Trajectory Comparisons
- `trajectory_sample_1.png` through `trajectory_sample_5.png`

Maps showing:
- Past track (black line with dots)
- True future track (green line with triangles)
- Predicted track (red dashed line with stars)
- Start and end points
- Mean error in kilometers

### 3. Intensity Comparisons
- `intensity_sample_1.png` through `intensity_sample_5.png`

Time series plots showing:
- Past intensity history
- True future intensity
- Predicted intensity
- Forecast start point
- RMSE and MAE metrics

### 4. Satellite Imagery
- `frames_sample_1.png` through `frames_sample_5.png`

Grid showing multiple channels and timesteps:
- Past frames (historical observations)
- True future frames (ground truth)
- Predicted future frames

### 5. Animations
- `animation_sample_1.gif` through `animation_sample_5.gif`

Animated GIFs showing:
- Evolving satellite imagery
- Moving typhoon track
- Current position indicator
- Time step and intensity information

### 6. Error Statistics
- `error_statistics.png`

Histograms showing distribution of:
- Track errors (km) across all samples
- Intensity RMSE (m/s) across all samples
- Mean and median values

### 7. HTML Report
- `prediction_report.html`

Comprehensive report containing:
- Summary metrics
- All visualizations embedded
- Interactive layout
- Styled presentation

## Extending to Real Satellite Data

The current implementation uses synthetic satellite imagery. To use real satellite data:

### Option 1: NOAA GOES Data

```python
# Example integration (requires additional libraries)
from satpy import Scene
from datetime import datetime

def load_goes_data(time, lat, lon, size=(64, 64)):
    """Load GOES satellite data for specific time/location"""
    # Implementation details...
    pass
```

### Option 2: NASA Data

Access NASA's data through their APIs:
- https://earthdata.nasa.gov/
- MODIS, VIIRS, or other relevant instruments

### Option 3: ERA5 Reanalysis

```python
# Use ERA5 reanalysis data (requires cdsapi)
import cdsapi

c = cdsapi.Client()
c.retrieve('reanalysis-era5-pressure-levels', {...})
```

## Performance Metrics

The visualization system tracks:

- **Track Error**: Mean distance between predicted and actual positions (km)
- **Intensity RMSE**: Root mean square error of wind speed predictions (m/s)
- **Intensity MAE**: Mean absolute error of wind speed predictions (m/s)

## Troubleshooting

### IBTrACS Data Download Issues

If automatic download fails:
1. Manually download from: https://www.ncei.noaa.gov/products/international-best-track-archive
2. Save to: `data/raw/ibtracs_wp.csv`

### Memory Issues

For large datasets:
- Reduce `n_samples` in config
- Decrease `batch_size`
- Use smaller `image_size`

### Slow Training

- Enable CUDA if available: The code automatically uses GPU if detected
- Reduce number of epochs for testing
- Use smaller models (reduce `hidden_dim`)

## Citation

If using IBTrACS data, please cite:
```
Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010: 
The International Best Track Archive for Climate Stewardship (IBTrACS): 
Unifying tropical cyclone best track data. Bulletin of the American Meteorological Society, 91, 363-376.
```

## Future Enhancements

Potential improvements:
1. Real-time satellite data integration
2. Interactive web-based visualizations
3. 3D visualizations of atmospheric structure
4. Ensemble prediction visualizations
5. Uncertainty quantification displays
6. Comparison with operational forecast models

## Examples

See `visualizations/` directory after running `test_with_real_data.py` for example outputs.

Open `visualizations/prediction_report.html` in your browser for a comprehensive view of all results!

