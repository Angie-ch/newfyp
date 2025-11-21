# What To Do Now - Step by Step Guide

## Current Situation

✅ **Good News:**
- Your config is already set to use `data/processed_temporal_split` ✓
- You have 1,100 training samples ready
- Real ERA5 data exists in `data/era5/` ✓
- Training script is ready with verification code ✓

⚠️ **Issue Found:**
- The current processed dataset was generated with **SYNTHETIC** meteorological data
- The `dataset_info.json` shows: `"meteorological_data": "Synthetic"`
- You need to regenerate the dataset using **REAL ERA5** data

## Step-by-Step Action Plan

### Step 1: Regenerate Dataset with Real ERA5 Data

You need to regenerate the processed dataset using your real ERA5 data:

```powershell
# Activate virtual environment
.\pytorch_gpu\Scripts\Activate.ps1

# Regenerate dataset with REAL ERA5 data
python data/generate_data_by_year.py
```

This will:
- Load real ERA5 data from `data/era5/ERA5_*_26data/` directories
- Extract real ERA5 frames for each typhoon timestep
- Create new `.npz` files in `data/processed_temporal_split/`
- Update `dataset_info.json` to show `"meteorological_data": "ERA5"`

**Expected output:**
```
================================================================================
LOADING ERA5 REANALYSIS DATA FROM DAILY FILES
================================================================================
Using ERA5 data directory: data/era5
Found ERA5 data for years: [2018, 2019, 2020, 2021]
Loading ERA5 data for X storms...
[OK] Successfully loaded ERA5 data for X/Y storms
```

### Step 2: Verify Real ERA5 Data Was Used

After regeneration, check:

```powershell
# Check the dataset info
Get-Content data/processed_temporal_split/dataset_info.json
```

Look for: `"meteorological_data": "ERA5"` (not "Synthetic")

### Step 3: Start Training with Real ERA5 Data

Once the dataset is regenerated:

```powershell
# Activate virtual environment
.\pytorch_gpu\Scripts\Activate.ps1

# Start training (will show verification that it's using real ERA5 data)
python train_autoencoder.py --config configs/autoencoder_config.yaml
```

The training script will print:
```
================================================================================
VERIFYING REAL ERA5 DATA USAGE
================================================================================
Data directory: [path]/data/processed_temporal_split
Directory exists: True
Found [number] training samples in [path]/train/cases
✓ Using REAL ERA5 data from processed_temporal_split
================================================================================
```

### Step 4: Monitor Training in Real-Time

**Option A: View in same terminal** (if running directly)
- Output will show in real-time automatically

**Option B: View log file in real-time** (if running in background)
```powershell
# In a separate terminal
Get-Content autoencoder_training.log -Wait -Tail 30
```

**Option C: Use TensorBoard** (for visual metrics)
```powershell
.\pytorch_gpu\Scripts\Activate.ps1
tensorboard --logdir logs/autoencoder
```
Then open http://localhost:6006 in your browser

## Quick Command Summary

```powershell
# 1. Regenerate dataset with real ERA5
.\pytorch_gpu\Scripts\Activate.ps1
python data/generate_data_by_year.py

# 2. Verify it worked
Get-Content data/processed_temporal_split/dataset_info.json | Select-String "meteorological_data"

# 3. Start training
python train_autoencoder.py --config configs/autoencoder_config.yaml

# 4. Monitor (in another terminal if needed)
Get-Content autoencoder_training.log -Wait -Tail 30
```

## Expected Training Time

- **Dataset Regeneration**: ~30-60 minutes (depends on number of storms and ERA5 files)
- **Training**: ~3-4 hours for 50 epochs (on CPU), faster on GPU

## What to Watch For

✅ **Good Signs:**
- "Found ERA5 data for years: [2018, 2019, 2020, 2021]"
- "Successfully loaded ERA5 data for X/Y storms"
- "✓ Using REAL ERA5 data from processed_temporal_split"
- Training loss decreasing over epochs

⚠️ **Warning Signs:**
- "WARNING: No ERA5 data directories found"
- "meteorological_data": "Synthetic" in dataset_info.json
- All zeros in frames (would indicate data loading issue)

## Next Steps After Training

Once training completes:
1. Check best model: `checkpoints/autoencoder/best.pth`
2. Evaluate on test set
3. Train diffusion model (if needed)
4. Run inference/predictions

---

**Ready to start? Run Step 1 now!**
