# Joint Encoding Architecture for Typhoon Prediction

## 🎯 Overview

This document describes the **joint encoding architecture** where ERA5 atmospheric fields and IBTrACS track/intensity data are **encoded together into a unified latent space**, then **decoded separately** for prediction.

---

## 📊 Architecture Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

INPUT DATA (Aligned):
├── ERA5: (B, T, 40, 64, 64) - Atmospheric fields
└── IBTrACS: (B, T, 3) - [lat, lon, intensity]

                              ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: JOINT ENCODING                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ERA5 (40, 64, 64)              IBTrACS (3,)                            │
│       │                              │                                   │
│       │                              ↓                                   │
│       │                         MLP Embedder                             │
│       │                              ↓                                   │
│       │                    Spatial Projection (16, 64, 64)              │
│       │                              │                                   │
│       └──────────────┬───────────────┘                                   │
│                      │ Concatenate (56, 64, 64)                         │
│                      ↓                                                   │
│              CNN Encoder (ResBlocks + Attention)                        │
│                      ↓                                                   │
│              Unified Latent (8, 8, 8)                                   │
│                                                                          │
│  ✅ Both ERA5 and IBTrACS information encoded together!                 │
└─────────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: DIFFUSION                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Past Latents (8, 8, 8)  ──→  Conditioning                             │
│                                     │                                    │
│  Future Latents (8, 8, 8) ──→  Add Noise  ──→  Diffusion Model         │
│                                     │                                    │
│                                     ↓                                    │
│                         Denoising Process (T steps)                     │
│                                     ↓                                    │
│                         Clean Future Latents (8, 8, 8)                  │
│                                                                          │
│  ✅ Diffusion operates on unified latent containing all info!           │
└─────────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: SEPARATE DECODING                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│              Unified Latent (8, 8, 8)                                   │
│                      ↓                                                   │
│              CNN Decoder (Upsample + ResBlocks)                         │
│                      ↓                                                   │
│              Shared Features (64, 64, 64)                               │
│                      │                                                   │
│         ┌────────────┴────────────┐                                     │
│         ↓                         ↓                                      │
│    ERA5 Head                 IBTrACS Head                               │
│    (Conv2D)                  (Pool + MLP)                               │
│         ↓                         ↓                                      │
│    ERA5 (40, 64, 64)         IBTrACS (3,)                               │
│                                                                          │
│  ✅ Separate outputs for each modality!                                 │
└─────────────────────────────────────────────────────────────────────────┘

OUTPUT:
├── Predicted ERA5: (B, T_future, 40, 64, 64)
├── Predicted Track: (B, T_future, 2) - [lat, lon]
└── Predicted Intensity: (B, T_future,) - wind speed
```

---

## 🔑 Key Components

### 1. Joint Autoencoder (`models/autoencoder/joint_autoencoder.py`)

**Purpose**: Encode ERA5 + IBTrACS together, decode separately

**Architecture**:
```python
class JointAutoencoder:
    # ENCODER
    def encode(era5, track, intensity):
        # 1. Embed IBTrACS scalars to features
        ibtracs_feat = MLP(track, intensity)  # (B, 256)
        
        # 2. Project to spatial maps
        ibtracs_spatial = Unflatten(ibtracs_feat)  # (B, 16, 64, 64)
        
        # 3. Concatenate with ERA5
        joint = concat([era5, ibtracs_spatial])  # (B, 56, 64, 64)
        
        # 4. Encode through CNN
        latent = CNN_Encoder(joint)  # (B, 8, 8, 8)
        
        return latent
    
    # DECODER
    def decode(latent):
        # 1. Decode to shared features
        shared = CNN_Decoder(latent)  # (B, 64, 64, 64)
        
        # 2. Split into separate heads
        era5 = ERA5_Head(shared)  # (B, 40, 64, 64)
        ibtracs = IBTrACS_Head(shared)  # (B, 3)
        
        return era5, track, intensity
```

**Loss Function**:
```python
loss = era5_weight * MSE(era5_recon, era5_target) +
       track_weight * MSE(track_recon, track_target) +
       intensity_weight * MSE(intensity_recon, intensity_target)
```

### 2. Joint Diffusion Trainer (`training/trainers/joint_diffusion_trainer.py`)

**Key Difference**: Uses joint encoding instead of separate encoding

```python
# OLD (Separate encoding):
past_latents = autoencoder.encode(past_frames)  # Only ERA5
condition = {
    'past_latents': past_latents,
    'past_track': past_track,        # Passed separately
    'past_intensity': past_intensity  # Passed separately
}

# NEW (Joint encoding):
past_latents = autoencoder.encode(
    past_frames, past_track, past_intensity  # All together!
)
condition = {
    'past_latents': past_latents,  # Contains everything!
}
```

### 3. Joint Trajectory Predictor (`inference/joint_trajectory_predictor.py`)

**Complete inference pipeline**:

```python
predictor = JointTrajectoryPredictor(autoencoder, diffusion_model)

results = predictor.predict(
    past_frames,      # (B, T_past, 40, 64, 64)
    past_track,       # (B, T_past, 2)
    past_intensity    # (B, T_past,)
)

# Returns:
# - future_frames: (B, N_samples, T_future, 40, 64, 64)
# - future_track: (B, N_samples, T_future, 2)
# - future_intensity: (B, N_samples, T_future)
```

---

## 🚀 Usage

### Training

#### Step 1: Train Joint Autoencoder
```bash
python train_joint_pipeline.py \
    --stage autoencoder \
    --config configs/joint_autoencoder.yaml
```

#### Step 2: Train Diffusion Model
```bash
python train_joint_pipeline.py \
    --stage diffusion \
    --config configs/joint_diffusion.yaml \
    --autoencoder_checkpoint checkpoints/joint_autoencoder/best.pth
```

#### Or train both sequentially:
```bash
python train_joint_pipeline.py \
    --stage both \
    --config configs/joint_pipeline.yaml
```

### Evaluation

```bash
python evaluate_joint_model.py \
    --config configs/joint_diffusion.yaml \
    --autoencoder_checkpoint checkpoints/joint_autoencoder/best.pth \
    --diffusion_checkpoint checkpoints/joint_diffusion/best.pth \
    --test_data data/processed/test/cases \
    --output_dir results/evaluation \
    --num_samples 10
```

### Inference

```python
from inference.joint_trajectory_predictor import JointTrajectoryPredictor

# Load predictor
predictor = JointTrajectoryPredictor.from_checkpoints(
    autoencoder_path='checkpoints/joint_autoencoder/best.pth',
    diffusion_path='checkpoints/joint_diffusion/best.pth',
    config=config
)

# Predict trajectory with uncertainty
results = predictor.predict_trajectory(
    past_frames,
    past_track,
    past_intensity,
    num_future_steps=12,
    num_samples=10  # Ensemble for uncertainty
)

# Access predictions
track_mean = results['track_mean']  # (T, 2)
track_std = results['track_std']    # (T, 2) - uncertainty
intensity_mean = results['intensity_mean']  # (T,)
```

---

## ✅ Advantages of Joint Encoding

### 1. **Unified Representation**
- Track and intensity information embedded in latent space alongside atmospheric patterns
- Model learns correlations between atmospheric state and typhoon characteristics

### 2. **Better Information Fusion**
- Encoder learns to fuse ERA5 and IBTrACS during compression
- Diffusion model operates on richer, more informative latents

### 3. **Consistency**
- Single latent space ensures consistency between atmospheric fields and track/intensity
- Reduces risk of contradictory predictions

### 4. **End-to-End Learning**
- Entire system learns joint representations from scratch
- Gradients flow through both modalities during training

### 5. **Simplified Conditioning**
- Diffusion model only needs past latents (no separate track/intensity conditioning)
- Cleaner architecture, fewer hyperparameters

---

## 📈 Expected Performance

### Reconstruction Quality (Autoencoder)
- **ERA5 MSE**: < 0.01 (normalized)
- **Track MAE**: < 0.5° (lat/lon)
- **Intensity MAE**: < 2 m/s

### Prediction Accuracy (Full Pipeline)
- **24h Track Error**: < 100 km
- **48h Track Error**: < 200 km
- **Intensity MAE**: < 5 m/s

---

## 🔧 Configuration

### Joint Autoencoder Config
```yaml
model:
  era5_channels: 40
  latent_channels: 8
  hidden_dims: [64, 128, 256, 256]
  use_attention: true

training:
  era5_weight: 1.0
  track_weight: 10.0      # Higher weight for track
  intensity_weight: 5.0   # Higher weight for intensity
```

### Diffusion Config
```yaml
model:
  latent_channels: 8
  hidden_dim: 256
  use_physics: true
  use_spiral_attention: true

training:
  timesteps: 1000
  loss_weights:
    diffusion: 1.0
    track: 0.5
    intensity: 0.3
    physics: 0.2
```

---

## 📝 Implementation Details

### Data Flow

**Training**:
1. Load batch: ERA5 + IBTrACS (aligned timestamps)
2. Joint encode: Both modalities → unified latent
3. Add noise: Diffusion forward process
4. Predict: Diffusion model denoises
5. Decode: Separate ERA5 and IBTrACS
6. Compute loss: Multi-task loss on all outputs

**Inference**:
1. Encode past observations jointly
2. Initialize noise
3. Iterative denoising (DDIM or full diffusion)
4. Decode final latents separately
5. Return predictions with uncertainty

### Memory Requirements

- **Autoencoder**: ~50M parameters
- **Diffusion Model**: ~100M parameters
- **Batch Size**: 4-8 (depends on GPU memory)
- **Latent Size**: 8 × 8 × 8 (320× compression from 40 × 64 × 64)

---

## 🐛 Troubleshooting

### Issue: NaN losses during autoencoder training
**Solution**: 
- Reduce learning rate
- Increase track/intensity loss weights
- Check data normalization

### Issue: Poor track reconstruction
**Solution**:
- Increase `track_weight` in loss
- Add more capacity to IBTrACS embedder
- Check spatial projection quality

### Issue: Diffusion model not converging
**Solution**:
- Ensure autoencoder is well-trained first
- Reduce diffusion learning rate
- Increase number of diffusion timesteps

---

## 📚 References

- Joint autoencoder: `models/autoencoder/joint_autoencoder.py`
- Training pipeline: `train_joint_pipeline.py`
- Inference: `inference/joint_trajectory_predictor.py`
- Evaluation: `evaluate_joint_model.py`

---

## 🎓 Citation

If you use this joint encoding architecture, please cite:

```bibtex
@article{typhoon_joint_encoding,
  title={Joint Encoding of Atmospheric Fields and Track Data for Typhoon Prediction},
  author={Your Name},
  year={2025}
}
```

