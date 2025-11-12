# Quick Start: Joint Encoding for Typhoon Prediction

This guide will get you started with the joint encoding architecture in 5 minutes!

---

## 🎯 What is Joint Encoding?

**Joint Encoding** means ERA5 atmospheric data and IBTrACS track/intensity data are:
1. ✅ **Encoded together** into a unified latent space
2. ✅ **Diffused together** in the latent space
3. ✅ **Decoded separately** back to ERA5 and IBTrACS

**Benefits**:
- Better information fusion
- Unified representation learning
- More consistent predictions

---

## 📦 Installation

```bash
# Clone repository
cd /Volumes/data/fyp/typhoon_prediction

# Install dependencies (if not already done)
pip install torch torchvision numpy pandas xarray netCDF4 matplotlib pyyaml tensorboard tqdm
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Data

Make sure your data is organized as:
```
data/processed/
├── train/cases/
│   ├── 2020_WP01_w00.npz
│   ├── 2020_WP01_w01.npz
│   └── ...
├── val/cases/
│   └── ...
└── test/cases/
    └── ...
```

Each `.npz` file should contain:
- `past_frames`: (T_past, C, H, W) ERA5 data
- `future_frames`: (T_future, C, H, W) ERA5 data
- `track_past`: (T_past, 2) track coordinates
- `track_future`: (T_future, 2) track coordinates
- `intensity_past`: (T_past,) wind speeds
- `intensity_future`: (T_future,) wind speeds

### Step 2: Train Models

#### Option A: Train Both Models Sequentially (Recommended)
```bash
python train_joint_pipeline.py \
    --stage both \
    --config configs/joint_pipeline.yaml \
    --device cuda
```

#### Option B: Train Step-by-Step

**Step 2a: Train Joint Autoencoder**
```bash
python train_joint_pipeline.py \
    --stage autoencoder \
    --config configs/joint_autoencoder.yaml \
    --device cuda
```

**Step 2b: Train Diffusion Model**
```bash
python train_joint_pipeline.py \
    --stage diffusion \
    --config configs/joint_diffusion.yaml \
    --autoencoder_checkpoint checkpoints/joint_autoencoder/best.pth \
    --device cuda
```

### Step 3: Evaluate

```bash
python evaluate_joint_model.py \
    --config configs/joint_diffusion.yaml \
    --autoencoder_checkpoint checkpoints/joint_autoencoder/best.pth \
    --diffusion_checkpoint checkpoints/joint_diffusion/best.pth \
    --test_data data/processed/test/cases \
    --output_dir results/evaluation \
    --num_samples 10
```

---

## 💻 Python API Usage

### Quick Prediction

```python
import torch
from inference.joint_trajectory_predictor import JointTrajectoryPredictor

# Load predictor
predictor = JointTrajectoryPredictor.from_checkpoints(
    autoencoder_path='checkpoints/joint_autoencoder/best.pth',
    diffusion_path='checkpoints/joint_diffusion/best.pth',
    config={
        'era5_channels': 40,
        'latent_channels': 8,
        'hidden_dim': 256,
        'num_heads': 8,
        'output_frames': 12,
        'timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02
    },
    device='cuda'
)

# Prepare input data
past_frames = torch.randn(1, 8, 40, 64, 64).cuda()      # Past ERA5
past_track = torch.randn(1, 8, 2).cuda()                # Past track
past_intensity = torch.randn(1, 8).cuda()               # Past intensity

# Predict with uncertainty quantification
results = predictor.predict_trajectory(
    past_frames,
    past_track,
    past_intensity,
    num_future_steps=12,
    num_samples=10  # Generate 10 ensemble members
)

# Access predictions
print(f"Predicted track (mean): {results['track_mean'].shape}")  # (12, 2)
print(f"Track uncertainty (std): {results['track_std'].shape}")  # (12, 2)
print(f"Predicted intensity: {results['intensity_mean'].shape}") # (12,)
print(f"Intensity uncertainty: {results['intensity_std'].shape}") # (12,)
```

### Detailed Prediction

```python
# Generate multiple samples for uncertainty
results = predictor.predict(
    past_frames,
    past_track,
    past_intensity,
    num_future_steps=12,
    num_samples=20,      # More samples = better uncertainty estimate
    ddim_steps=50,       # Faster sampling (50 steps instead of 1000)
    guidance_scale=1.0   # Conditioning strength
)

# Results contain:
# - future_frames: (1, 20, 12, 40, 64, 64) - Predicted ERA5 fields
# - future_track: (1, 20, 12, 2) - Predicted trajectories
# - future_intensity: (1, 20, 12) - Predicted intensities
# - latents: (1, 20, 12, 8, 8, 8) - Latent codes

# Compute ensemble statistics
import numpy as np

tracks = results['future_track'][0].cpu().numpy()  # (20, 12, 2)
track_mean = tracks.mean(axis=0)  # (12, 2)
track_std = tracks.std(axis=0)    # (12, 2)

# Visualize uncertainty
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i in range(20):
    plt.plot(tracks[i, :, 1], tracks[i, :, 0], 'b-', alpha=0.2)
plt.plot(track_mean[:, 1], track_mean[:, 0], 'r-', linewidth=3, label='Mean')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory Ensemble')
plt.legend()
plt.grid(True)
plt.savefig('trajectory_ensemble.png')
```

---

## 📊 Monitoring Training

### TensorBoard

```bash
# Monitor autoencoder training
tensorboard --logdir logs/joint_autoencoder

# Monitor diffusion training
tensorboard --logdir logs/joint_diffusion
```

### Key Metrics to Watch

**Autoencoder**:
- `Loss/train_era5`: ERA5 reconstruction loss (should decrease to < 0.01)
- `Loss/train_track`: Track reconstruction loss (should decrease to < 0.5)
- `Loss/train_intensity`: Intensity reconstruction loss (should decrease to < 2.0)

**Diffusion**:
- `Train/total`: Total training loss
- `Train/diffusion`: Diffusion loss (noise prediction)
- `Train/track`: Track prediction loss
- `Train/intensity`: Intensity prediction loss
- `Train/physics`: Physics constraint loss

---

## 🎛️ Configuration

### Minimal Config

```yaml
# configs/my_config.yaml
data:
  data_dir: "data/processed"
  past_frames: 8
  future_frames: 12
  era5_channels: 40

model:
  era5_channels: 40
  latent_channels: 8
  hidden_dim: 256

training:
  epochs: 50
  batch_size: 8
  learning_rate: 1.0e-4
```

### Advanced Config

See `configs/joint_pipeline.yaml` for full options including:
- Loss weights
- Physics constraints
- Attention mechanisms
- Diffusion schedule
- EMA settings

---

## 🔍 Understanding the Architecture

### Visual Overview

```
┌──────────────────────────────────────────────────────┐
│  INPUT: ERA5 (40, 64, 64) + IBTrACS (3,)            │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  JOINT ENCODER                                       │
│  - Embed IBTrACS to spatial (16, 64, 64)            │
│  - Concatenate with ERA5 → (56, 64, 64)             │
│  - CNN Encoder → Latent (8, 8, 8)                   │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  DIFFUSION MODEL                                     │
│  - Condition on past latents                         │
│  - Denoise future latents                            │
│  - Physics-informed constraints                      │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  SEPARATE DECODER                                    │
│  - Shared decoder backbone                           │
│  - ERA5 Head → (40, 64, 64)                         │
│  - IBTrACS Head → (3,) [lat, lon, intensity]        │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  OUTPUT: Predicted ERA5 + Track + Intensity          │
└──────────────────────────────────────────────────────┘
```

### Key Files

- **Joint Autoencoder**: `models/autoencoder/joint_autoencoder.py`
- **Diffusion Model**: `models/diffusion/physics_diffusion.py`
- **Training**: `train_joint_pipeline.py`
- **Inference**: `inference/joint_trajectory_predictor.py`
- **Evaluation**: `evaluate_joint_model.py`

---

## 🐛 Common Issues

### Issue: Out of Memory
**Solution**: Reduce batch size in config
```yaml
training:
  batch_size: 4  # or even 2
```

### Issue: Slow Training
**Solution**: Use fewer workers or reduce data loading
```yaml
training:
  num_workers: 2  # reduce from 4
```

### Issue: Poor Reconstruction
**Solution**: Train autoencoder longer or adjust loss weights
```yaml
training:
  epochs: 100  # increase from 50
  track_weight: 20.0  # increase if track reconstruction is poor
```

---

## 📈 Expected Training Time

On a single NVIDIA A100 GPU:
- **Joint Autoencoder**: ~4-6 hours (50 epochs)
- **Diffusion Model**: ~12-16 hours (100 epochs)
- **Total**: ~20 hours

On a single NVIDIA RTX 3090:
- **Joint Autoencoder**: ~8-10 hours
- **Diffusion Model**: ~24-30 hours
- **Total**: ~35 hours

---

## 📚 Next Steps

1. ✅ Read full documentation: `docs/JOINT_ENCODING_ARCHITECTURE.md`
2. ✅ Experiment with hyperparameters in configs
3. ✅ Visualize predictions with `evaluate_joint_model.py`
4. ✅ Try different loss weights for your use case
5. ✅ Implement custom evaluation metrics

---

## 🆘 Getting Help

- **Documentation**: See `docs/JOINT_ENCODING_ARCHITECTURE.md`
- **Examples**: Check `configs/` for configuration examples
- **Code**: All code is well-commented

---

## 🎉 You're Ready!

You now have a complete pipeline for typhoon trajectory prediction using joint encoding!

**Start training**:
```bash
python train_joint_pipeline.py --stage both --config configs/joint_pipeline.yaml
```

Good luck! 🌪️

