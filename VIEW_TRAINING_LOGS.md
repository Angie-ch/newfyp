# View Training Logs in Real-Time

## Quick Commands

### Option 1: View Existing Log File (Real-Time)
```powershell
# View last 50 lines and follow new entries
Get-Content autoencoder_training.log -Wait -Tail 50
```

### Option 2: Start New Training with Real-Time Output
```powershell
# Activate virtual environment and run training
.\pytorch_gpu\Scripts\Activate.ps1
python train_autoencoder.py --config configs/autoencoder_config.yaml
```

### Option 3: Save Training Output to File and View
```powershell
# Start training with output saved to file
.\pytorch_gpu\Scripts\Activate.ps1
python train_autoencoder.py --config configs/autoencoder_config.yaml 2>&1 | Tee-Object -FilePath training_live.log

# In another terminal, view the log in real-time:
Get-Content training_live.log -Wait -Tail 30
```

### Option 4: Use Python Script
```powershell
python show_training_logs.py
```

## Current Training Status

Based on the log file `autoencoder_training.log`:

- **Data Source**: `data/processed` (Note: Should be `data/processed_temporal_split` for real ERA5 data)
- **Training Samples**: 63
- **Validation Samples**: 12
- **Epochs**: 50
- **Device**: CPU
- **Model Parameters**: 11.33M

## Important Note

The current log shows training from `data/processed` directory. To ensure you're using **real ERA5 data**, make sure the config points to `data/processed_temporal_split`:

```yaml
data:
  data_dir: "data/processed_temporal_split"  # ← Should be this for real ERA5 data
```

## View TensorBoard Logs

For visual training metrics:
```powershell
.\pytorch_gpu\Scripts\Activate.ps1
tensorboard --logdir logs/autoencoder
```

Then open http://localhost:6006 in your browser.











