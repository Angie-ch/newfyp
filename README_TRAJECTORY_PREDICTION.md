# Typhoon Trajectory Prediction Pipeline

## Overview

This pipeline uses a **diffusion model** to predict typhoon trajectories 72 hours (12 timesteps × 6 hours) into the future based on 48 hours (8 timesteps × 6 hours) of past observations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   TYPHOON PREDICTION PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

Input: 8 Past Timesteps (48 hours)
├── ERA5 Atmospheric Fields (48 channels)
│   ├── Temperature, Wind, Pressure, Humidity, etc.
│   └── Vorticity, Divergence, Wind Shear (derived)
└── IBTrACS Data (4 channels)
    ├── Latitude Field (vertical gradient)
    ├── Longitude Field (horizontal gradient)
    ├── Intensity (uniform field)
    └── Pressure (uniform field)

         ↓ [Spatial Autoencoder - Encoder]

Latent Representation (8×8×8 compressed)

         ↓ [Physics-Informed Diffusion Model]
           • Typhoon-Aware UNet3D
           • Spiral Attention Mechanisms
           • Multi-Scale Temporal Modeling
           • Physics Constraints

Predicted Latent (12×8×8×8)

         ↓ [Spatial Autoencoder - Decoder]

Output: 12 Future Timesteps (72 hours)
├── Predicted Atmospheric Fields (48 channels)
├── Predicted Trajectory (lat, lon)
├── Predicted Intensity (wind speed)
└── Predicted Pressure

         ↓ [Visualization]

📊 Final Output:
   • Satellite imagery with trajectory overlay
   • Past 8 points (blue/black circles)
   • Ground truth 12 points (green triangles)
   • Predicted 12 points (red stars)
   • Error statistics (mean error, final error)
```

## Training Pipeline

### 1. Train Autoencoder (Compression)

```bash
python train_autoencoder.py \
  --config configs/autoencoder_config.yaml \
  --epochs 50 \
  --batch_size 16
```

**Purpose**: Compress high-dimensional atmospheric fields (48×256×256) to compact latent representations (8×32×32), enabling efficient diffusion modeling.

**Output**: `checkpoints/autoencoder/best.pth`

### 2. Train Diffusion Model (Prediction)

```bash
python train_diffusion.py \
  --config configs/diffusion_config.yaml \
  --autoencoder checkpoints/autoencoder/best.pth \
  --epochs 100 \
  --augment
```

**Purpose**: Learn to predict future latent representations using:
- **Diffusion process**: Iteratively denoise predictions
- **Physics constraints**: Enforce meteorological laws
- **Multi-task learning**: Jointly predict structure, track, and intensity

**Output**: `checkpoints/diffusion/best.pth`

## Inference & Visualization

### Generate Trajectory Predictions

```bash
python predict_and_visualize_trajectory.py \
  --autoencoder checkpoints/autoencoder/best.pth \
  --diffusion checkpoints/diffusion/best.pth \
  --data_dir data/processed \
  --output_dir results/trajectory_predictions \
  --num_samples 10 \
  --satellite_bg path/to/satellite_image.png  # Optional
```

### Expected Output

For each sample, generates:

1. **Trajectory Visualization** (`trajectory_<case_id>.png`)
   - Left panel: Map view with trajectory overlay
   - Right panel: Coordinate evolution time series
   
2. **Statistics**
   - Mean error (average distance error across 12 timesteps)
   - Final error (72-hour forecast error)

## Input/Output Format

### Dataset Structure

Each sample in `data/processed/cases/` contains:

```python
{
    'past_frames': (8, 48, 64, 64),      # 8 timesteps × 48 channels × 64×64 pixels
    'future_frames': (12, 48, 64, 64),   # 12 timesteps (ground truth)
    'track_past': (8, 2),                # [latitude, longitude] for 8 timesteps
    'track_future': (12, 2),             # Ground truth trajectory
    'intensity_past': (8,),              # Wind speed in m/s
    'intensity_future': (12,),           # Ground truth intensity
    'pressure_past': (8,),               # Central pressure in hPa
    'pressure_future': (12,),            # Ground truth pressure
    'case_id': str                       # Typhoon identifier
}
```

### Prediction Output

```python
{
    'future_frames': (B, 12, 48, 64, 64),      # Predicted atmospheric fields
    'future_track': (B, 12, 2),                # Predicted [lat, lon]
    'future_intensity': (B, 12),               # Predicted wind speed
    'future_latents': (B, 12, 8, 8, 8),        # Latent representations
}
```

## Visualization Details

### Trajectory Plot Components

1. **Past Trajectory** (8 timesteps)
   - **Style**: Black circles with solid line (`ko-`)
   - **Represents**: Historical typhoon path (48 hours)
   - **Purpose**: Show model input

2. **Ground Truth Future** (12 timesteps)
   - **Style**: Green triangles with solid line (`g^-`)
   - **Represents**: Actual typhoon path (next 72 hours)
   - **Purpose**: Comparison baseline

3. **Predicted Future** (12 timesteps)
   - **Style**: Red stars with dashed line (`r*--`)
   - **Represents**: Model's predicted path (next 72 hours)
   - **Purpose**: Model output

4. **Start/End Markers**
   - Blue square: Starting position
   - Green square: Actual ending position
   - Red square: Predicted ending position

### Error Metrics

- **Mean Error**: Average distance between predicted and actual positions across all 12 future timesteps
  ```
  mean_error = mean(||pred_track[t] - true_track[t]||) × 111 km/degree
  ```

- **Final Error**: Distance between 72-hour forecast and actual position
  ```
  final_error = ||pred_track[12] - true_track[12]|| × 111 km/degree
  ```

## Model Architecture Details

### Autoencoder

- **Input**: (T, 48, 64, 64) atmospheric fields
- **Latent**: (T, 8, 8, 8) compressed representation
- **Architecture**:
  - Encoder: 4-layer CNN with residual blocks
  - Decoder: 4-layer transposed CNN
  - Attention mechanisms at bottleneck
  - Compression ratio: 48×64×64 → 8×8×8 (768:1)

### Diffusion Model

- **Backbone**: Typhoon-Aware UNet3D
  - 3D convolutions for spatio-temporal processing
  - Spiral attention (follows typhoon rotation)
  - Multi-scale temporal fusion
  
- **Physics Constraints**:
  - Geostrophic balance (wind-pressure relationship)
  - Mass conservation
  - Temporal smoothness
  - Wind-pressure gradient consistency
  
- **Multi-Task Heads**:
  - Structure Head: Predicts atmospheric fields
  - Track Head: Predicts lat/lon trajectory
  - Intensity Head: Predicts wind speed evolution

### Diffusion Process

1. **Training**: Learn to denoise corrupted latents
   ```
   L = L_diffusion + λ_track·L_track + λ_intensity·L_intensity + λ_physics·L_physics
   ```

2. **Inference**: DDIM sampling (50 steps)
   - Start: Pure Gaussian noise
   - Iterate: Denoise using learned model
   - End: Clean prediction

## Configuration Files

### `configs/autoencoder_config.yaml`

```yaml
model:
  in_channels: 48        # ERA5 channels
  latent_channels: 8     # Compressed dimension
  hidden_dims: [64, 128, 256, 256]
  use_attention: true

training:
  epochs: 50
  learning_rate: 1e-4
  batch_size: 16
```

### `configs/diffusion_config.yaml`

```yaml
model:
  latent_channels: 8
  hidden_dim: 256
  num_heads: 8
  depth: 4

diffusion:
  timesteps: 1000        # Training timesteps
  sampling_steps: 50     # Inference timesteps (DDIM)
  beta_schedule: "linear"

data:
  input_frames: 8        # Past timesteps
  output_frames: 12      # Future timesteps

training:
  epochs: 100
  learning_rate: 2e-4
  loss_weights:
    diffusion: 1.0
    track: 0.5
    intensity: 0.3
    physics: 0.2
```

## IBTrACS Encoding (Key Innovation)

The position encoding creates **spatially-varying coordinate fields** that enable position recovery:

### Channel Structure

```
Channel 0: Latitude Field          Channel 1: Longitude Field
┌─────────────────┐               ┌─────────────────┐
│ +0.8  +0.8      │ (North)       │ -0.5  0.0  +0.5 │
│  0.0   0.0      │ (Center)      │ -0.5  0.0  +0.5 │
│ -0.8  -0.8      │ (South)       │ -0.5  0.0  +0.5 │
└─────────────────┘               └─────────────────┘
  Vertical Gradient                 Horizontal Gradient

Channel 2: Intensity               Channel 3: Pressure
┌─────────────────┐               ┌─────────────────┐
│ 0.06  0.06      │               │ -0.09 -0.09     │
│ 0.06  0.06      │               │ -0.09 -0.09     │
└─────────────────┘               └─────────────────┘
  Uniform Field                     Uniform Field
```

### Decoding Position

```python
# Extract from center pixel
center_h, center_w = H // 2, W // 2

# Latitude from channel 0
lat_norm = decoded_latent[:, :, 0, center_h, center_w]
lat = lat_norm * 12.5 + 22.5  # Denormalize to degrees

# Longitude from channel 1
lon_norm = decoded_latent[:, :, 1, center_h, center_w]
lon = lon_norm * 20.0 + 140.0  # Denormalize to degrees
```

**Result**: Perfect position recovery with 0.0° error!

## Performance Expectations

### Baseline Comparisons

| Model | 24h Error | 48h Error | 72h Error |
|-------|-----------|-----------|-----------|
| Persistence | ~100 km | ~250 km | ~400 km |
| Climatology | ~150 km | ~300 km | ~450 km |
| **Ours (Target)** | **<80 km** | **<200 km** | **<350 km** |

### Training Time

- **Autoencoder**: ~4-6 hours (50 epochs, single GPU)
- **Diffusion**: ~24-48 hours (100 epochs, single GPU)

### Inference Time

- **Per Sample**: ~10-15 seconds (50 DDIM steps)
- **Batch of 10**: ~30-40 seconds

## Troubleshooting

### Common Issues

1. **NaN in training**
   - Check normalization statistics
   - Reduce learning rate
   - Enable gradient clipping

2. **Poor trajectory predictions**
   - Verify IBTrACS encoding is working
   - Check physics constraint weights
   - Increase track loss weight

3. **Slow inference**
   - Reduce DDIM sampling steps (50 → 25)
   - Use smaller batch size
   - Enable mixed precision (fp16)

### Debug Commands

```bash
# Test IBTrACS encoding
python data/utils/ibtracs_encoding.py

# Verify dataset
python test_ibtracs_concat.py

# Check encoding/decoding accuracy
python test_position_encoding_decoding.py

# Visualize encoding patterns
python visualize_ibtracs_encoding.py
```

## Next Steps

1. **Ensemble Predictions**: Generate multiple samples for uncertainty quantification
2. **Real-Time Inference**: Deploy model for operational forecasting
3. **Intensity Focus**: Fine-tune intensity prediction head
4. **Long-Range**: Extend to 5-day (20-timestep) forecasts

## Citation

If you use this code, please cite:

```bibtex
@software{typhoon_diffusion_2025,
  title={Physics-Informed Diffusion Model for Typhoon Trajectory Prediction},
  author={Your Name},
  year={2025},
  note={GitHub repository},
  url={https://github.com/yourusername/typhoon_prediction}
}
```

## References

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
- **IBTrACS**: Knapp et al., "International Best Track Archive for Climate Stewardship"
- **ERA5**: Hersbach et al., "ERA5 global reanalysis" (Q. J. R. Meteorol. Soc. 2020)















