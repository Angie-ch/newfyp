# Temporal Configuration: 8 Past + 12 Future

## Overview

The typhoon prediction model uses **8 timesteps of historical data** to predict **12 timesteps into the future**.

## Rationale

### Why 8 Past Timesteps?
- **Computational efficiency**: Shorter input sequences = faster training
- **Recent context**: Last 8 hours (or 48 hours at 6h intervals) captures the most relevant recent dynamics
- **Sufficient for patterns**: Typhoon structure and movement patterns are well-captured in 8 frames

### Why 12 Future Timesteps?
- **Longer forecast horizon**: More useful for disaster preparedness and warnings
- **Standard practice**: 72-hour forecasts (12 × 6h) are industry standard
- **Challenging task**: Tests model's ability to extrapolate far into future

## Data Dimensions

### Training Data Format
```
Sample structure:
├── past_frames: (8, 48, 64, 64)
│   └── 8 timesteps × 48 channels × 64×64 spatial
├── future_frames: (12, 48, 64, 64)
│   └── 12 timesteps × 48 channels × 64×64 spatial
├── track_past: (8, 2)
│   └── 8 timesteps × (lat, lon)
├── track_future: (12, 2)
│   └── 12 timesteps × (lat, lon)
├── intensity_past: (8,)
│   └── 8 timesteps of wind speed
└── intensity_future: (12,)
    └── 12 timesteps of wind speed
```

### Model Architecture
```
Input (Conditioning):
├── past_latents: (B, 8, C_latent, H/8, W/8)
├── past_track: (B, 8, 2)
└── past_intensity: (B, 8)

Output (Prediction):
├── future_latents: (B, 12, C_latent, H/8, W/8)
├── track: (B, 12, 2)
└── intensity: (B, 12)
```

## Time Intervals

### Option 1: 6-hour intervals (standard)
- **Past coverage**: 8 × 6h = 48 hours (2 days)
- **Future coverage**: 12 × 6h = 72 hours (3 days)
- **Total duration**: 120 hours (5 days)
- **Use case**: Standard operational forecasting

### Option 2: 1-hour intervals (high-resolution)
- **Past coverage**: 8 × 1h = 8 hours
- **Future coverage**: 12 × 1h = 12 hours
- **Total duration**: 20 hours
- **Use case**: Rapid intensification studies, short-term nowcasting

## Configuration Files

### 1. `run_real_data_pipeline.py`
```python
samples = loader.create_dataset(
    past_timesteps=8,        # 8 historical frames
    future_timesteps=12      # 12 future frames to predict
)
```

### 2. `preprocess_era5_1h.py`
```python
config = {
    'input_frames': 8,       # 8 hours of past data
    'output_frames': 12,     # 12 hours of future prediction
    'time_interval_hours': 1
}
```

### 3. Model initialization
```python
model = PhysicsInformedDiffusionModel(
    latent_channels=8,
    output_frames=12  # Predict 12 future timesteps
)
```

## Impact on Existing Data

⚠️ **Important**: If you have existing preprocessed data with **12 past + 8 future**, you need to:

1. **Delete old data**:
   ```bash
   rm -rf data/processed/cases/*
   rm data/processed/statistics.json
   ```

2. **Regenerate with new configuration**:
   ```bash
   python run_real_data_pipeline.py --n-samples 100
   ```

3. **Retrain all models** (old checkpoints are incompatible)

## Benefits of This Configuration

### ✅ Advantages
1. **Longer predictions**: 72-hour forecasts vs 48-hour
2. **More useful**: Better for operational forecasting
3. **Efficient**: Less input data to process
4. **Challenging**: Tests model's extrapolation ability

### ⚠️ Considerations
1. **Less history**: May miss longer-term patterns
2. **Harder task**: Predicting further into future increases uncertainty
3. **Data requirements**: Storms need sufficient future data (12+ timesteps)

## Validation

After regenerating data, verify dimensions:
```python
import numpy as np

# Load a sample
data = np.load('data/processed/cases/case_0000.npz')

# Check shapes
assert data['past_frames'].shape[0] == 8, "Should have 8 past frames"
assert data['future_frames'].shape[0] == 12, "Should have 12 future frames"
assert data['track_past'].shape[0] == 8
assert data['track_future'].shape[0] == 12

print("✓ Data dimensions are correct!")
```

## References

- IBTrACS data is typically at 6-hour intervals
- ERA5 data can be interpolated to 1-hour intervals
- Standard forecast horizon: 24h (short), 72h (medium), 120h+ (extended)

