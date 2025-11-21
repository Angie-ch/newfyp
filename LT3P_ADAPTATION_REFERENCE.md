# LT3P Adaptation: Using IBTrACS + ERA5 Instead of UM Data

## Reference Paper

**Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data**  
Park et al., ICLR 2024 Spotlight  
Repository: https://github.com/iclr2024submit/LT3P

## Key Differences: LT3P vs. Our Approach

### LT3P Original Approach
- **Data Source**: Unified Model (UM) forecast data (real-time operational forecasts)
- **Rationale**: Avoids reanalysis data (like ERA5) because:
  - Reanalysis data is not available in real-time
  - Requires time for adjustment/calibration
  - Not suitable for operational forecasting
- **Advantage**: Can be used for real-time predictions
- **Limitation**: UM forecast data may have systematic biases

### Our Adapted Approach
- **Data Source**: IBTrACS (best-track) + ERA5 (reanalysis)
- **Rationale**: 
  - ERA5 provides the most accurate historical representation of weather conditions
  - Better for research, model development, and retrospective analysis
  - IBTrACS provides ground-truth typhoon tracks and intensity
- **Advantage**: Higher data quality for training and evaluation
- **Use Case**: Research, model development, and post-event analysis

## Architecture Similarities

Both approaches share similar core components:

### 1. Physics-Conditioned Model
- ✅ **Our Implementation**: `PhysicsInformedDiffusionModel` with `PhysicsProjector`
- ✅ **LT3P**: Physics-conditioned approach
- **Key**: Both enforce physical constraints (geostrophic balance, mass conservation, etc.)

### 2. Multi-Task Learning
- ✅ **Our Implementation**: 
  - Structure prediction (atmospheric fields)
  - Track prediction (lat/lon coordinates)
  - Intensity prediction (wind speed)
- ✅ **LT3P**: Similar multi-task approach
- **Key**: Both predict trajectory and intensity simultaneously

### 3. Long-Term Prediction
- ✅ **Our Implementation**: 72-hour prediction horizon (12 timesteps × 6 hours)
- ✅ **LT3P**: 72-hour prediction horizon
- **Key**: Both target operational forecasting timeframes

## Data Adaptation Details

### Input Data Format

#### LT3P (UM Data)
```
- UM forecast fields (operational weather model output)
- Real-time availability
- May include forecast uncertainty
```

#### Our Approach (ERA5 + IBTrACS)
```python
# ERA5 Reanalysis Data (48 channels)
- 7 variables × 4 pressure levels = 28 base channels
  * Geopotential (z)
  * Temperature (t)
  * U/V wind components (u, v)
  * Relative humidity (r)
  * Specific humidity (q)
  * Vorticity (vo)
- Additional derived variables
- Spatial: 64×64 grid (cropped around typhoon)
- Temporal: 6-hourly resolution

# IBTrACS Best-Track Data
- Position (lat, lon) - ground truth
- Maximum sustained wind speed
- Minimum central pressure
- Storm category/classification
```

### Data Preprocessing

#### Our Implementation
```python
# data/generate_data_by_year.py
1. Load IBTrACS typhoon tracks
2. For each typhoon timestep:
   - Extract ERA5 data from daily files
   - Crop 64×64 region around typhoon center
   - Stack multiple timesteps (past + future)
3. Create training samples with:
   - past_frames: (T_past, 48, 64, 64) ERA5 data
   - future_frames: (T_future, 48, 64, 64) ERA5 data
   - track_past/future: (T, 2) lat/lon coordinates
   - intensity_past/future: (T,) wind speeds
```

#### Key Differences
- **LT3P**: Uses UM forecast data directly (may need bias correction)
- **Our Approach**: Uses ERA5 reanalysis (most accurate historical data)
- **Both**: Extract regions around typhoon center
- **Both**: Use 6-hourly temporal resolution

## Model Architecture Comparison

### Core Components

| Component | LT3P | Our Implementation | Status |
|-----------|------|-------------------|--------|
| Physics Constraints | ✅ | ✅ `PhysicsProjector` | Implemented |
| Multi-Task Heads | ✅ | ✅ `TrackHead`, `IntensityHead`, `StructureHead` | Implemented |
| Temporal Modeling | ✅ | ✅ `MultiScaleTemporalBlock` | Implemented |
| Attention Mechanism | ✅ | ✅ `SpiralAttention` (typhoon-aware) | Implemented |
| Diffusion Process | ✅ | ✅ DDPM with DDIM sampling | Implemented |

### Architecture Details

#### Our Physics-Informed Diffusion Model
```python
# models/diffusion/physics_diffusion.py
class PhysicsInformedDiffusionModel:
    - TyphoonAwareUNet3D (backbone)
    - StructureHead (atmospheric fields)
    - TrackHead (trajectory)
    - IntensityHead (wind speed)
    - PhysicsProjector (constraints)
```

#### Key Innovations (Both Approaches)
1. **Physics Conditioning**: Enforce atmospheric physics laws
2. **Multi-Task Learning**: Joint prediction of structure, track, and intensity
3. **Long-Term Horizon**: 72-hour predictions
4. **Temporal Modeling**: Multi-scale temporal features

## Training Differences

### LT3P Training
- Uses UM forecast data (may include forecast errors)
- Trained on operational forecast scenarios
- May need bias correction for UM data

### Our Training
- Uses ERA5 reanalysis (most accurate historical representation)
- Trained on historical typhoon events
- Ground-truth tracks from IBTrACS
- Better for model development and evaluation

## Evaluation Metrics

Both approaches use similar metrics:

### Track Prediction
- Mean Absolute Error (MAE) in degrees
- Great Circle Distance (GCD)
- 24/48/72-hour forecast errors

### Intensity Prediction
- Mean Absolute Error (MAE) in m/s
- Category classification accuracy

### Structure Prediction
- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)

## Implementation Status

### ✅ Completed Components
- [x] Physics-informed diffusion model
- [x] Multi-task prediction heads
- [x] Typhoon-aware architecture (spiral attention)
- [x] Multi-scale temporal modeling
- [x] ERA5 data processing pipeline
- [x] IBTrACS data integration
- [x] Training infrastructure
- [x] Evaluation metrics

### 🔄 In Progress
- [ ] Regeneration with real ERA5 data (currently showing "Synthetic")
- [ ] Model training with ERA5 data
- [ ] Performance evaluation

## Usage Notes

### When to Use Each Approach

**Use LT3P (UM Data)** when:
- Real-time operational forecasting is required
- Need immediate predictions without waiting for reanalysis
- Building operational forecast systems

**Use Our Approach (ERA5 + IBTrACS)** when:
- Research and model development
- Retrospective analysis
- Need highest data quality for training
- Evaluating model performance against ground truth

## Citation

If using this adaptation, please cite both:

```bibtex
@inproceedings{park2023long,
  title={Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data},
  author={Park, Young-Jae and Seo, Minseok and Kim, Doyi and Kim, Hyeri and Choi, Sanghoon and Choi, Beomkyu and Ryu, Jeongwon and Son, Sohee and Jeon, Hae-Gon and Choi, Yeji},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}

@article{your_work_2024,
  title={Long-Term Typhoon Prediction with ERA5 Reanalysis and IBTrACS Data},
  author={Your Name},
  journal={Your Journal},
  year={2024},
  note={Adapted from LT3P using ERA5+IBTrACS instead of UM data}
}
```

## References

- LT3P Repository: https://github.com/iclr2024submit/LT3P
- ERA5 Data: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- IBTrACS Data: https://www.ncei.noaa.gov/products/international-best-track-archive


