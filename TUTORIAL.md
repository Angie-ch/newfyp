# Typhoon Prediction Pipeline Tutorial

## Complete Guide to Training and Evaluation

This tutorial walks you through the entire pipeline from data preprocessing to model evaluation.

---

## Table of Contents

1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Training Autoencoder](#training-autoencoder)
4. [Training Diffusion Model](#training-diffusion-model)
5. [Inference & Evaluation](#inference--evaluation)
6. [Ablation Studies](#ablation-studies)
7. [Troubleshooting](#troubleshooting)

---

## 1. Setup

### Installation

```bash
# Create conda environment
conda create -n typhoon python=3.10
conda activate typhoon

# Install dependencies
cd typhoon_prediction
pip install -r requirements.txt
```

### Directory Structure

```
typhoon_prediction/
├── data/
│   ├── raw/            # Raw ERA5 and IBTrACS data
│   └── processed/      # Preprocessed data
├── checkpoints/        # Model checkpoints
│   ├── autoencoder/
│   └── diffusion/
├── logs/              # Training logs
└── results/           # Evaluation results
```

---

## 2. Data Preparation

### Download Data

**ERA5 Reanalysis Data:**
- Visit: https://cds.climate.copernicus.eu/
- Download variables: u/v wind, temperature, geopotential, humidity, pressure
- Pressure levels: 1000, 925, 850, 700, 500, 300 hPa
- Time resolution: 6-hourly
- Region: Western Pacific (typhoon-prone area)

**IBTrACS Data:**
- Visit: https://www.ncdc.noaa.gov/ibtracs/
- Download: `IBTrACS.WP.v04r00.csv` (Western Pacific basin)

### Preprocess Data

```bash
python preprocess_data.py \
    --era5_dir data/raw/era5 \
    --ibtracs data/raw/IBTrACS.WP.v04r00.csv \
    --output data/processed \
    --start_date 2015-01-01 \
    --end_date 2020-12-31 \
    --compute_stats \
    --stats_samples 100
```

**Parameters:**
- `--input_frames`: Number of past frames (default: 12 = 3 days)
- `--output_frames`: Number of future frames (default: 8 = 2 days)
- `--time_interval`: Time between frames in hours (default: 6)
- `--min_intensity`: Minimum wind speed in m/s (default: 17.0 = Tropical Storm)

**Output:**
- `data/processed/*.pkl`: Preprocessed typhoon cases
- `data/processed/normalization_stats.pkl`: Global statistics

---

## 3. Training Autoencoder

### Configuration

Edit `configs/autoencoder_config.yaml`:

```yaml
model:
  in_channels: 40      # ERA5 variables
  latent_channels: 8   # Compression factor
  hidden_dims: [64, 128, 256, 256]
  use_attention: true

training:
  epochs: 50
  learning_rate: 1.0e-4
  batch_size: 16
```

### Train

```bash
python train_autoencoder.py \
    --config configs/autoencoder_config.yaml \
    --augment
```

**Expected Training Time:**
- GPU (V100): ~6-8 hours
- GPU (A100): ~3-4 hours

**Monitoring:**

```bash
tensorboard --logdir logs/autoencoder
```

**Expected Performance:**
- Reconstruction MSE: < 0.05
- PSNR: > 30 dB

### Validation

The autoencoder should compress 256×256 frames to 32×32 latents (8× compression) with minimal information loss.

---

## 4. Training Diffusion Model

### Configuration

Edit `configs/diffusion_config.yaml`:

```yaml
model:
  latent_channels: 8
  hidden_dim: 256
  num_heads: 8
  use_spiral_attention: true
  use_multiscale_temporal: true

diffusion:
  timesteps: 1000
  beta_schedule: "linear"

training:
  epochs: 100
  learning_rate: 2.0e-4
  
  loss_weights:
    diffusion: 1.0
    track: 0.5
    intensity: 0.3
    physics: 0.2
    consistency: 0.1
  
  physics_weights:
    geostrophic: 1.0
    mass_conservation: 0.1
    temporal_smooth: 0.1
    wind_pressure: 0.5
```

### Train

```bash
python train_diffusion.py \
    --config configs/diffusion_config.yaml \
    --autoencoder checkpoints/autoencoder/best.pth \
    --augment
```

**Expected Training Time:**
- GPU (V100): ~2-3 days
- GPU (A100): ~1-1.5 days

**Monitoring:**

```bash
tensorboard --logdir logs/diffusion
```

Monitor:
- `Loss/train` and `Loss/val`: Should decrease steadily
- `Train/diffusion`: Noise prediction loss
- `Train/track`: Track prediction loss (km)
- `Train/intensity`: Intensity prediction loss (m/s)
- `Train/physics_total`: Physics constraint losses

---

## 5. Inference & Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --autoencoder checkpoints/autoencoder/best.pth \
    --diffusion checkpoints/diffusion/best.pth \
    --data data/processed \
    --output results/evaluation.json \
    --sampling_steps 50
```

**Parameters:**
- `--sampling_steps`: DDIM steps (50 = faster, 1000 = highest quality)
- `--batch_size`: Evaluation batch size

### Expected Results

**Track Error (48h forecast):**
- Persistence baseline: ~300 km
- Our model: ~120 km (60% improvement)

**Intensity MAE:**
- Persistence baseline: ~15 m/s
- Our model: ~5 m/s (67% improvement)

**Physics Consistency:**
- Baseline: ~60% pass rate
- Our model: ~95% pass rate

### Visualize Results

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results/evaluation.json', 'r') as f:
    results = json.load(f)

# Plot track error vs lead time
lead_times = [6, 12, 18, 24, 30, 36, 42, 48]
track_errors = [results[f'track_error_{t}h_km'] for t in lead_times]

plt.figure(figsize=(10, 6))
plt.plot(lead_times, track_errors, marker='o', linewidth=2)
plt.xlabel('Lead Time (hours)')
plt.ylabel('Track Error (km)')
plt.title('Track Prediction Error vs Lead Time')
plt.grid(True)
plt.savefig('results/track_error.png', dpi=300)
```

---

## 6. Ablation Studies

To validate each innovation, train models with components disabled:

### Baseline (No Innovations)

```yaml
# configs/diffusion_baseline.yaml
model:
  use_spiral_attention: false
  use_multiscale_temporal: false

training:
  loss_weights:
    physics: 0.0  # Disable physics losses
    consistency: 0.0
```

### + Physics Only

```yaml
training:
  loss_weights:
    physics: 0.2
    consistency: 0.0
```

### + Spiral Attention

```yaml
model:
  use_spiral_attention: true
  use_multiscale_temporal: false
```

### + Multi-Scale Temporal

```yaml
model:
  use_spiral_attention: true
  use_multiscale_temporal: true
```

### Full Model

Use default `diffusion_config.yaml`

**Compare Results:**

```bash
python compare_ablations.py \
    --baseline results/baseline_evaluation.json \
    --physics results/physics_evaluation.json \
    --spiral results/spiral_evaluation.json \
    --multiscale results/multiscale_evaluation.json \
    --full results/full_evaluation.json \
    --output results/ablation_comparison.png
```

---

## 7. Troubleshooting

### Common Issues

**Issue 1: Out of Memory**

Solution:
```yaml
# Reduce batch size
data:
  batch_size: 4  # Instead of 16

# Or reduce model size
model:
  hidden_dim: 128  # Instead of 256
```

**Issue 2: Training Instability**

Solution:
```yaml
# Increase gradient clipping
training:
  gradient_clip: 0.5  # Lower value

# Or reduce learning rate
training:
  learning_rate: 1.0e-4  # Instead of 2.0e-4
```

**Issue 3: Poor Track Predictions**

- Check track normalization in data preprocessing
- Increase track loss weight
- Verify IBTrACS data quality

**Issue 4: Physics Violations**

- Increase physics loss weight
- Check autoencoder reconstruction quality
- Verify ERA5 data preprocessing

### Performance Tips

**Faster Training:**
- Use mixed precision: `torch.cuda.amp`
- Reduce DDIM sampling steps during validation
- Use gradient checkpointing for large models

**Better Performance:**
- Train longer (100+ epochs)
- Increase model capacity (hidden_dim = 512)
- Use more data (longer time range)
- Fine-tune loss weights

---

## Summary

**Complete Pipeline:**

1. ✅ Download ERA5 + IBTrACS data
2. ✅ Preprocess data → `data/processed/`
3. ✅ Train autoencoder (~8 hours)
4. ✅ Train diffusion model (~2 days)
5. ✅ Evaluate on test set
6. ✅ Run ablation studies
7. ✅ Visualize & analyze results

**Key Innovations:**

1. **Physics-Informed Diffusion**: Constrained predictions
2. **Typhoon-Aware Architecture**: Spiral attention + multi-scale temporal
3. **Multi-Task Learning**: Structure + track + intensity

**Expected Improvements:**

- Track error: 60% better than persistence
- Intensity MAE: 67% better than persistence
- Physics consistency: 95% pass rate

---

## Questions?

Check:
- `README.md` for overview
- `models/` for architecture details
- `training/` for training implementation
- GitHub Issues for community support

**Good luck with your research! 🌪️🚀**

