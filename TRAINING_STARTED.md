# Training Pipeline Started

## Summary

I've set up and tested the complete typhoon prediction training pipeline. All modules have been tested and are working correctly.

## What Was Done

### 1. Module Testing ✅
Created `test_pipeline_modules.py` that tests each component incrementally:
- ✅ Data loading
- ✅ Autoencoder forward/backward pass
- ✅ Diffusion model forward/backward pass
- ✅ Training loops with small batches

### 2. Fixed Issues
- Fixed dataset interface mismatch (removed `past_frames`/`future_frames` parameters)
- Fixed channel mismatch (data has 48 channels, model uses 40 - now slices correctly)
- Fixed import paths
- Fixed diffusion model input format

### 3. Training Scripts
- `train_joint_pipeline.py` - Main training script (fixed)
- `start_training.sh` - Convenience script to run full pipeline
- `test_pipeline_modules.py` - Module testing script

## How to Run Training

### Option 1: Run Full Pipeline (Recommended)
```bash
./start_training.sh
```

### Option 2: Run Stages Separately

**Stage 1: Train Autoencoder**
```bash
python train_joint_pipeline.py \
    --stage autoencoder \
    --config configs/joint_autoencoder.yaml \
    --device cuda  # or cpu
```

**Stage 2: Train Diffusion Model**
```bash
python train_joint_pipeline.py \
    --stage diffusion \
    --config configs/joint_diffusion.yaml \
    --device cuda  # or cpu
```

### Option 3: Test First (Recommended)
```bash
python test_pipeline_modules.py
```

## Configuration

- **Autoencoder Config**: `configs/joint_autoencoder.yaml`
  - 40 ERA5 channels (uses first 40 of 48 available)
  - 8 latent channels
  - 50 epochs
  - Batch size: 8

- **Diffusion Config**: `configs/joint_diffusion.yaml`
  - Uses pretrained autoencoder
  - 100 epochs
  - Batch size: 4
  - 1000 diffusion timesteps

## Data Structure

- Training: 63 samples (after filtering NaN)
- Validation: 12 samples
- Data location: `data/processed/`
- Each sample has:
  - 12 past frames (72 hours)
  - 8 future frames (48 hours)
  - Track data (lat, lon)
  - Intensity data (wind speed)

## Notes

- The model uses 40 channels even though data has 48 (first 40 are used)
- Training on CPU will be slow - use GPU if available
- Checkpoints are saved every 5 epochs
- Best model is saved as `best.pth` in checkpoint directories

## Next Steps

1. Run `./start_training.sh` to begin training
2. Monitor logs in `logs/` directories
3. Check checkpoints in `checkpoints/` directories
4. After training, use inference scripts to generate predictions

