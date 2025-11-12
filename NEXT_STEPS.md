# Next Steps - Typhoon Prediction Project

## 🔄 Currently Running

### Autoencoder Training
The autoencoder is currently training and will take approximately **3-4 hours** to complete 50 epochs.

**Current Status**: Epoch 2/50 (running smoothly)
- Train Loss: 896.87 → 311 (improving!)
- Val Loss: 311.13
- No NaN issues ✓

**Monitor Training**:
```bash
# Watch live progress
tail -f /Volumes/data/fyp/typhoon_prediction/autoencoder_training.log

# Check current status
tail -20 /Volumes/data/fyp/typhoon_prediction/autoencoder_training.log

# View checkpoints
ls -lh /Volumes/data/fyp/typhoon_prediction/checkpoints/autoencoder/
```

**What happens during training**:
- Every epoch (~4 minutes): Trains on 63 samples, validates on 12 samples
- Every 5 epochs: Saves checkpoint to `checkpoints/autoencoder/checkpoint_epoch_X.pth`
- When val loss improves: Saves best model to `checkpoints/autoencoder/best.pth`

---

## 📋 After Autoencoder Completes

### Step 1: Start Diffusion Training (~6-8 hours)

Once autoencoder reaches epoch 50/50:

```bash
cd /Volumes/data/fyp/typhoon_prediction

# Start diffusion training
nohup python train_diffusion.py \
    --config configs/diffusion_config.yaml \
    --autoencoder_checkpoint checkpoints/autoencoder/best.pth \
    > diffusion_training.log 2>&1 &

# Monitor progress
tail -f diffusion_training.log
```

**What this does**:
- Loads the trained autoencoder (frozen)
- Trains diffusion model in latent space for 100 epochs
- Learns to predict future latent states
- Trains track, intensity, and pressure prediction heads
- Applies physics constraints during training

---

### Step 2: Run Inference

After diffusion training completes:

```bash
cd /Volumes/data/fyp/typhoon_prediction

# Generate predictions on test set
python inference.py \
    --autoencoder_checkpoint checkpoints/autoencoder/best.pth \
    --diffusion_checkpoint checkpoints/diffusion/best.pth \
    --autoencoder_config configs/autoencoder_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --data_dir data/processed \
    --output_dir results/predictions \
    --split test \
    --batch_size 4
```

**Output**:
- Saves predictions to `results/predictions/predictions.npz`
- Contains predicted atmospheric fields, tracks, intensity, and pressure
- Includes ground truth for comparison

---

### Step 3: Evaluate Results

After inference completes:

```bash
cd /Volumes/data/fyp/typhoon_prediction

# Evaluate predictions
python evaluate.py \
    --predictions results/predictions/predictions.npz \
    --output_dir results/evaluation
```

**Output**:
- Computes metrics:
  - Track error (km)
  - Intensity error (m/s)
  - Pressure error (hPa)
  - Field reconstruction error (MSE, MAE)
  
- Generates visualizations:
  - Track comparison plots
  - Intensity comparison plots
  - Error evolution over forecast time
  
- Creates evaluation report: `results/evaluation/evaluation_report.txt`

---

## 🚨 If Training Stops or Encounters Issues

### Check if Training is Running
```bash
ps aux | grep "train_autoencoder\|train_diffusion" | grep -v grep
```

### Restart Autoencoder Training
```bash
cd /Volumes/data/fyp/typhoon_prediction

# Start from scratch
python train_autoencoder.py --config configs/autoencoder_config.yaml

# Or resume from checkpoint (if available)
python train_autoencoder.py \
    --config configs/autoencoder_config.yaml \
    --resume checkpoints/autoencoder/checkpoint_epoch_X.pth
```

### Restart Diffusion Training
```bash
cd /Volumes/data/fyp/typhoon_prediction

python train_diffusion.py \
    --config configs/diffusion_config.yaml \
    --autoencoder_checkpoint checkpoints/autoencoder/best.pth
```

### Common Issues

1. **Out of Memory**:
   - Reduce batch size in config files
   - Autoencoder: `data.batch_size: 8` (currently 16)
   - Diffusion: `data.batch_size: 4` (currently 8)

2. **Training Too Slow**:
   - Reduce number of epochs
   - Autoencoder: `training.epochs: 20` (currently 50)
   - Diffusion: `training.epochs: 50` (currently 100)

3. **NaN Loss Returns**:
   - Lower learning rate further
   - Check that NaN filtering is working
   - Verify data normalization

---

## 📊 Expected Results

### Training Time
- **Autoencoder**: ~3-4 hours (50 epochs on CPU)
- **Diffusion**: ~6-8 hours (100 epochs on CPU)
- **Total**: ~10-12 hours

### Model Performance
Based on similar typhoon prediction models:

**Track Prediction**:
- 24-hour forecast: 50-100 km error
- 48-hour forecast: 100-200 km error

**Intensity Prediction**:
- MAE: 3-5 m/s
- RMSE: 5-8 m/s

**Field Reconstruction**:
- MSE: < 1.0 (normalized)
- Visual quality: Good for 24-48 hour forecasts

---

## 🎯 Success Criteria

Training is successful if:

1. ✅ **Autoencoder**:
   - Val loss decreases over epochs
   - Final val loss < 100
   - No NaN/Inf values
   - Best model saved

2. ✅ **Diffusion**:
   - All losses (diffusion, track, intensity, physics) decrease
   - Can generate coherent predictions
   - No NaN/Inf values
   - Best model saved

3. ✅ **Inference**:
   - Generates predictions for all test samples
   - Predictions are physically reasonable
   - No NaN/Inf in outputs

4. ✅ **Evaluation**:
   - Track errors < 300 km (48-hour forecast)
   - Intensity MAE < 10 m/s
   - Visualizations show reasonable predictions

---

## 📁 Key Files Created

### Scripts (Ready to Use)
- ✅ `train_autoencoder.py` - Training script
- ✅ `train_diffusion.py` - Diffusion training
- ✅ `inference.py` - Generate predictions
- ✅ `evaluate.py` - Compute metrics and plots

### Documentation
- ✅ `README.md` - Project overview
- ✅ `TRAINING_PIPELINE_SUMMARY.md` - Architecture details
- ✅ `PROGRESS_SUMMARY.md` - What we've accomplished
- ✅ `NEXT_STEPS.md` - This file (step-by-step instructions)

### Models
- ✅ `models/autoencoder/autoencoder.py` - Encoder + Decoder
- ✅ `models/diffusion/physics_diffusion.py` - Diffusion model
- ✅ `models/components/blocks.py` - Building blocks

### Data
- ✅ `data/processed/cases/` - 93 clean samples
- ✅ `data/processed/normalization_stats.pkl` - Statistics
- ✅ `data/era5/` - 1,574 ERA5 files
- ✅ `data/raw/ibtracs_wp.csv` - Track database

### Configs
- ✅ `configs/autoencoder_config.yaml` - Hyperparameters
- ✅ `configs/diffusion_config.yaml` - Hyperparameters

---

## 🔍 Monitoring Progress

### Check Autoencoder Progress
```bash
# Quick status
tail -5 /Volumes/data/fyp/typhoon_prediction/autoencoder_training.log

# See if still running
ps aux | grep train_autoencoder | grep -v grep

# View TensorBoard (if available)
tensorboard --logdir /Volumes/data/fyp/typhoon_prediction/logs/autoencoder
```

### Check Diffusion Progress
```bash
# Quick status
tail -5 /Volumes/data/fyp/typhoon_prediction/diffusion_training.log

# See if still running
ps aux | grep train_diffusion | grep -v grep

# View TensorBoard
tensorboard --logdir /Volumes/data/fyp/typhoon_prediction/logs/diffusion
```

---

## ⏰ Estimated Timeline

**Now**: Autoencoder training (Epoch 2/50)
- **+3 hours**: Autoencoder completes
- **+3-9 hours**: Start and complete diffusion training
- **+9-10 hours**: Run inference and evaluation
- **+10 hours**: Complete project with results!

---

## 📞 Questions?

If you encounter any issues:

1. Check the log files (`autoencoder_training.log`, `diffusion_training.log`)
2. Verify the process is still running (`ps aux | grep train`)
3. Review `PROGRESS_SUMMARY.md` for troubleshooting
4. Check data integrity (`python check_data_nan.py` if needed)

---

## 🎉 When Everything is Done

You will have:

1. ✅ Trained autoencoder model (compresses 48-channel atmospheric data)
2. ✅ Trained diffusion model (predicts future states)
3. ✅ Predictions for test set (tracks, intensity, pressure, fields)
4. ✅ Evaluation metrics and visualizations
5. ✅ Complete typhoon prediction pipeline

**Final outputs**:
- `checkpoints/autoencoder/best.pth` - Trained autoencoder
- `checkpoints/diffusion/best.pth` - Trained diffusion model
- `results/predictions/predictions.npz` - All predictions
- `results/evaluation/` - Metrics and plots
- `results/evaluation/evaluation_report.txt` - Performance summary

---

Good luck! The training is running smoothly. Just let it continue and follow these steps when it completes. 🚀

