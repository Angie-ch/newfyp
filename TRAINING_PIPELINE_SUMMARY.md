# Typhoon Prediction Training Pipeline Summary

## Architecture Overview

### 1. Autoencoder (Encoder + Decoder Train Together)

**Purpose**: Learn compressed latent representations of atmospheric fields

**Components**:
- **Encoder**: Compresses (B, 48, 64, 64) → (B, 8, 8, 8)
  - 4 downsampling stages with ResBlocks
  - Attention at bottleneck for global context
  - Reduces spatial dimensions by factor of 8
  
- **Decoder**: Reconstructs (B, 8, 8, 8) → (B, 48, 64, 64) 
  - 4 upsampling stages with ResBlocks
  - Symmetric architecture to encoder
  - Reconstructs atmospheric fields

**Training**:
- **Loss**: MSE(reconstruction, original)
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 with cosine schedule
- **Epochs**: 50
- **Batch Size**: 16

**Parameters**: 11.86M

### 2. Diffusion Model (Uses Frozen Autoencoder)

**Purpose**: Predict future latent states conditioned on past

**Architecture**:
- Operates in 8-channel latent space (compressed by autoencoder)
- U-Net with temporal attention
- Physics-informed constraints:
  - Geostrophic balance
  - Mass conservation
  - Wind-pressure relationship
  - Temporal smoothness
  
**Multi-task Heads**:
- Track prediction (lat/lon coordinates)
- Intensity prediction (max wind speed)
- Central pressure prediction

**Training**:
- Uses frozen autoencoder (encoder/decoder not updated)
- **Loss Components**:
  - Diffusion loss: 1.0
  - Track loss: 0.5
  - Intensity loss: 0.3
  - Physics loss: 0.2
  - Consistency loss: 0.1
- **Optimizer**: AdamW
- **Learning Rate**: 2e-4
- **Epochs**: 100
- **Diffusion Steps**: 1000 (training), 50 (inference with DDIM)

## Data Pipeline

### Input Data
- **ERA5**: 48 channels of atmospheric variables
  - 7 variables × 4 pressure levels = 28 base channels
  - Additional derived variables
  - Spatial: 64×64 grid
  - Temporal: 6-hourly resolution
  
- **IBTrACS**: Typhoon tracks and intensity
  - Position (lat, lon)
  - Maximum sustained wind
  - Minimum central pressure
  - Storm category

### Processed Samples
- **Past Frames**: 12 timesteps (72 hours)
- **Future Frames**: 8 timesteps (48 hours) 
- **Total Duration**: 120 hours (5 days) per sample

### Dataset Split
- Train: 70 samples (70%)
- Val: 15 samples (15%)
- Test: 15 samples (15%)

## Training Status

### Current Progress

**Autoencoder Training**:
- ✓ Started: Currently training
- Status: Epoch 1/50
- Device: CPU
- Log: `autoencoder_training.log`
- Checkpoints: `checkpoints/autoencoder/`

**Diffusion Model**:
- ⏳ Pending (starts after autoencoder completes)
- Will use best autoencoder checkpoint

## Inference Pipeline

### Steps:
1. **Encode Past Frames**:
   ```
   past_latents = encoder(past_frames)  # (B, 12, 8, 8, 8)
   ```

2. **Diffusion Prediction**:
   ```
   future_latents = diffusion.sample(past_latents)  # (B, 8, 8, 8, 8)
   track, intensity = diffusion.predict_track_intensity(past_latents)
   ```

3. **Decode Future Frames**:
   ```
   future_frames = decoder(future_latents)  # (B, 8, 48, 64, 64)
   ```

### Outputs:
- Future atmospheric fields (8 timesteps ahead)
- Predicted track (lat/lon for 8 timesteps)
- Predicted intensity (wind speed for 8 timesteps)
- Predicted pressure (central pressure for 8 timesteps)

## Key Files

### Code:
- `models/autoencoder/autoencoder.py` - Encoder + Decoder model
- `models/diffusion/physics_diffusion.py` - Diffusion model
- `training/trainers/autoencoder_trainer.py` - AE training logic
- `training/trainers/diffusion_trainer.py` - Diffusion training logic
- `data/datasets/typhoon_dataset.py` - Data loading
- `inference.py` - Prediction pipeline

### Configs:
- `configs/autoencoder_config.yaml` - AE hyperparameters
- `configs/diffusion_config.yaml` - Diffusion hyperparameters

### Data:
- `data/processed/cases/` - 100 preprocessed samples
- `data/processed/metadata.csv` - Sample information
- `data/era5/` - Raw ERA5 data (4 years, 1574 files)
- `data/raw/ibtracs_wp.csv` - IBTrACS typhoon database

### Outputs:
- `checkpoints/` - Trained model weights
- `logs/` - Training logs and metrics
- `results/` - Predictions and visualizations

## Next Steps

1. ✓ **Preprocessing**: Completed (100 samples ready)
2. 🔄 **Train Autoencoder**: In progress (Epoch 1/50)
3. ⏳ **Train Diffusion**: Starts after AE completes
4. ⏳ **Inference**: Generate predictions on test set
5. ⏳ **Evaluation**: Compute metrics and visualizations

## Notes

- **Why train encoder+decoder together?**  
  They form one autoencoder model that learns to compress and reconstruct data. The reconstruction loss ensures the latent space preserves important atmospheric information.

- **Why freeze autoencoder during diffusion training?**  
  The latent space is already learned. Freezing prevents it from changing during diffusion training, ensuring stable latent representations.

- **1-hour interpolation attempt**:  
  We attempted to create 1-hour resolution data from ERA5, but encountered issues with varying spatial dimensions and missing files. The existing 6-hourly data (100 samples) is sufficient for demonstration.

## Training Time Estimates

- **Autoencoder** (CPU): ~2-4 hours for 50 epochs
- **Diffusion** (CPU): ~6-12 hours for 100 epochs
- **Total Pipeline**: ~10-16 hours on CPU

With GPU, this would be 10-20x faster.

## Monitoring Training

Check progress:
```bash
# Autoencoder training log
tail -f autoencoder_training.log

# Training curves
ls -lh logs/autoencoder/

# Checkpoints
ls -lh checkpoints/autoencoder/
```

## Questions Addressed

**Q: Do we have to train the autodecoder also?**

A: Yes! The "autoencoder" consists of both encoder and decoder, and they train **together** as one model. The training process:

1. Input frame → **Encoder** → latent representation
2. Latent → **Decoder** → reconstructed frame  
3. Loss = MSE(reconstructed, original)
4. Both encoder and decoder weights update via backpropagation

This is different from training them separately. They learn together so the encoder produces latents that the decoder can accurately reconstruct.

During diffusion training, both encoder and decoder are frozen and used as feature extractors/generators.


