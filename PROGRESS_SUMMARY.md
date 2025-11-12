# Typhoon Prediction Project - Progress Summary

## ✅ Completed Tasks

### 1. Data Preprocessing ✓
- **ERA5 Data**: Downloaded and processed 4 years (2020-2023) of atmospheric data
  - 1,574 ERA5 files
  - 7 variables × 4 pressure levels = 28 base channels
  - Additional derived variables (vorticity, divergence, etc.) → 48 total channels
  - Spatial resolution: 64×64 grid
  - Temporal resolution: 6-hourly

- **IBTrACS Data**: Processed Western Pacific typhoon tracks
  - Storm positions (lat, lon)
  - Maximum sustained wind speed
  - Minimum central pressure
  - Storm category

- **Processed Samples**: Created 100 typhoon cases
  - Each case: 12 past timesteps (72 hours) + 8 future timesteps (48 hours)
  - Total: 120 hours (5 days) per sample
  - Format: `.npz` files with past_frames, future_frames, track data, intensity

- **Data Split**:
  - Training: 63 samples (after filtering NaN)
  - Validation: 12 samples
  - Test: 18 samples

### 2. Model Architecture ✓

#### Autoencoder
- **Purpose**: Compress 48-channel atmospheric fields spatially
- **Architecture**:
  - Encoder: (B, 48, 64, 64) → (B, 8, 8, 8)
  - Decoder: (B, 8, 8, 8) → (B, 48, 64, 64)
  - 4 stages of downsampling/upsampling
  - Residual blocks with GroupNorm and SiLU
  - Optional attention at bottleneck (currently disabled for stability)
- **Parameters**: 11.33M

#### Diffusion Model
- **Purpose**: Predict future atmospheric states in latent space
- **Architecture**:
  - U-Net with temporal attention
  - Operates on 8-channel latents
  - Physics-informed constraints:
    - Geostrophic balance
    - Mass conservation  
    - Wind-pressure relationships
    - Temporal smoothness
- **Multi-task heads**:
  - Track prediction (lat/lon)
  - Intensity prediction (max wind)
  - Pressure prediction (central pressure)
- **Training**: 1000 diffusion steps (50 steps for DDIM inference)

### 3. Training Pipeline ✓

#### Configuration Files
- `configs/autoencoder_config.yaml`: Autoencoder hyperparameters
- `configs/diffusion_config.yaml`: Diffusion hyperparameters

#### Training Scripts
- `train_autoencoder.py`: Trains encoder+decoder together
- `train_diffusion.py`: Trains diffusion with frozen autoencoder
- `training/trainers/autoencoder_trainer.py`: Training logic
- `training/trainers/diffusion_trainer.py`: Diffusion training logic

#### Key Fixes Applied
1. **Data Normalization**: Added proper normalization using precomputed stats
   - Normalizes base 28 channels using mean/std
   - Clips extreme values to [-10, 10]

2. **NaN Filtering**: Removed 7 corrupted samples
   - Checks all samples on load
   - Filters out samples with NaN values
   - Reduced dataset: 70 → 63 train, 15 → 12 val

3. **Model Initialization**: Added proper weight initialization
   - Kaiming initialization for Conv layers
   - Xavier initialization for output layer (gain=0.1)
   - Proper GroupNorm and Linear layer initialization

4. **Training Stability**:
   - Reduced learning rate: 1e-4 → 5e-5
   - Aggressive gradient clipping: 0.5
   - NaN/Inf detection with batch skipping
   - Gradient norm logging

5. **Attention Disabled**: Temporarily disabled for stability
   - Can re-enable after confirming stable training

## 🔄 Currently Running

### Autoencoder Training
- **Status**: Epoch 2/50 in progress
- **Device**: CPU
- **Training Time**: ~50-60 seconds per batch, ~4 minutes per epoch
- **Estimated Total Time**: ~3-4 hours for 50 epochs

#### Epoch 1 Results:
- Train Loss: 896.87
- Val Loss: 311.13
- Best model saved ✓

#### Training Configuration:
```yaml
Learning Rate: 5e-5
Weight Decay: 0.01
Gradient Clip: 0.5
Batch Size: 16
Optimizer: AdamW
Scheduler: Cosine Annealing
```

#### Monitoring:
```bash
# Watch training progress
tail -f autoencoder_training.log

# Check latest status
tail -20 autoencoder_training.log

# View checkpoints
ls -lh checkpoints/autoencoder/
```

## ⏳ Pending Tasks

### 1. Complete Autoencoder Training
- Let training run for 50 epochs (~3-4 hours)
- Monitor for:
  - Decreasing train/val loss
  - No NaN/Inf values
  - Validation loss improvement
- Expected outcome: Compressed latent representation that preserves atmospheric features

### 2. Train Diffusion Model
- Start after autoencoder completes
- Load best autoencoder checkpoint
- Freeze encoder/decoder weights
- Train diffusion in latent space for 100 epochs
- Estimated time: ~6-8 hours on CPU

### 3. Inference Pipeline
- Create `inference.py` script
- Load both trained models
- Generate predictions for test set
- Output:
  - Future atmospheric fields
  - Predicted tracks
  - Predicted intensity/pressure

### 4. Evaluation & Visualization
- Compute metrics:
  - Track error (distance in km)
  - Intensity error (wind speed RMSE)
  - Pressure error (RMSE)
  - Field reconstruction quality (SSIM, MSE)
- Create visualizations:
  - Predicted vs actual tracks
  - Predicted vs actual intensity
  - Atmospheric field comparisons
  - Error maps

## 📊 Current Status Summary

| Component | Status | Progress |
|-----------|--------|----------|
| Data Preprocessing | ✅ Complete | 100% |
| Data Cleaning (NaN removal) | ✅ Complete | 100% |
| Autoencoder Architecture | ✅ Complete | 100% |
| Diffusion Architecture | ✅ Complete | 100% |
| Training Scripts | ✅ Complete | 100% |
| Autoencoder Training | 🔄 Running | 2/50 epochs (4%) |
| Diffusion Training | ⏳ Pending | 0% |
| Inference Script | ⏳ Pending | 0% |
| Evaluation | ⏳ Pending | 0% |

## 🐛 Issues Resolved

### Issue 1: NaN Loss During Training
**Problem**: Loss became NaN immediately after first batch

**Root Cause**: 7 samples in dataset contained NaN values from preprocessing

**Solution**: 
- Added `_filter_nan_samples()` method to dataset
- Checks each sample on load and filters out corrupted ones
- Result: Clean dataset with 63 train / 12 val samples

### Issue 2: Training Instability
**Problem**: Model produced NaN even with some good batches

**Root Causes**:
- Learning rate too high
- Improper weight initialization
- No data normalization

**Solutions**:
- Reduced learning rate: 5e-5
- Added proper weight initialization with Kaiming/Xavier
- Added data normalization with clipping to [-10, 10]
- Reduced gradient clipping: 0.5
- Added NaN detection and batch skipping

### Issue 3: 1-Hour Resolution Preprocessing Failed
**Problem**: Attempted to create 1-hour resolution data but encountered:
- Varying spatial dimensions across files
- Missing files in the downloaded data
- Complex interpolation requirements

**Solution**: Used existing 6-hourly data (sufficient for demonstration)

## 📁 Key Files

### Code
- `models/autoencoder/autoencoder.py` - Encoder+Decoder
- `models/diffusion/physics_diffusion.py` - Diffusion model
- `training/trainers/autoencoder_trainer.py` - AE training
- `training/trainers/diffusion_trainer.py` - Diffusion training
- `data/datasets/typhoon_dataset.py` - Data loading with NaN filtering
- `data/preprocessing/era5_processor.py` - ERA5 processing
- `data/preprocessing/ibtracs_processor.py` - Track processing
- `data/preprocessing/create_samples.py` - Sample creation

### Configs
- `configs/autoencoder_config.yaml` - AE hyperparameters
- `configs/diffusion_config.yaml` - Diffusion hyperparameters

### Data
- `data/era5/` - 1,574 ERA5 files (2020-2023)
- `data/raw/ibtracs_wp.csv` - IBTrACS database
- `data/processed/cases/` - 93 clean samples (.npz)
- `data/processed/normalization_stats.pkl` - Normalization statistics

### Outputs
- `checkpoints/autoencoder/best.pth` - Best autoencoder checkpoint
- `logs/autoencoder/` - TensorBoard logs
- `autoencoder_training.log` - Training output

## 🎯 Next Steps

1. **Monitor Current Training** (~3-4 hours)
   - Check log periodically
   - Ensure validation loss decreases
   - Verify no NaN/Inf issues

2. **Start Diffusion Training** (after AE completes)
   ```bash
   python train_diffusion.py --config configs/diffusion_config.yaml
   ```

3. **Create Inference Script**
   - Load trained models
   - Generate predictions
   - Save results

4. **Evaluate Performance**
   - Compute metrics
   - Create visualizations
   - Compare with baselines

## ⏱️ Time Estimates

- ✅ Data Preprocessing: ~2 hours (Complete)
- ✅ Model Development: ~3 hours (Complete)
- ✅ Debugging & Fixes: ~2 hours (Complete)
- 🔄 Autoencoder Training: ~3-4 hours (Running, 4% complete)
- ⏳ Diffusion Training: ~6-8 hours (Pending)
- ⏳ Inference & Evaluation: ~1-2 hours (Pending)

**Total Project Time**: ~17-21 hours (8 hours completed, 9-13 remaining)

## 📝 Notes

- Training is running on CPU (GPU would be 10-20x faster)
- Attention mechanism temporarily disabled for stability
- Can re-enable attention after confirming stable training
- 6-hourly resolution is sufficient for typhoon prediction
- Dataset size (63 samples) is small but adequate for demonstration

## 🔗 References

- ERA5 Reanalysis: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
- IBTrACS: https://www.ncei.noaa.gov/products/international-best-track-archive
- Diffusion Models: Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
- DDIM Sampling: Song et al. (2020), "Denoising Diffusion Implicit Models"

