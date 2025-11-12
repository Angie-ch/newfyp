# Critical Fixes Needed - Quick Reference

## Priority 1: Data Issues (MUST FIX FIRST)

### Issue 1: NaN in Statistics

**Problem:** `data/processed/statistics.json` has all NaN values

**Fix:**
```bash
# Run diagnostic script
python diagnose_data_issues.py

# This will:
# 1. Identify which samples have NaN
# 2. Identify which channels have NaN
# 3. Offer to recompute statistics
```

**Root Cause:** 
- Synthetic ERA5 data generation creates NaN in derived fields
- Need real ERA5 data OR fix synthetic data generation

### Issue 2: No Real ERA5 Data

**Problem:** 0/100 samples use real ERA5 data (all synthetic)

**Fix:**
1. Register at https://cds.climate.copernicus.eu/
2. Create API key: https://cds.climate.copernicus.eu/api-how-to
3. Install cdsapi: `pip install cdsapi`
4. Download ERA5 data:

```python
# Save as download_era5.py
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind',
            'temperature', 'geopotential',
            'relative_humidity', 'specific_humidity'
        ],
        'pressure_level': [
            '300', '500', '700',
            '850', '925', '1000'
        ],
        'year': '2021',
        'month': ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'],
        'day': [f'{i:02d}' for i in range(1, 32)],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [40, 110, 10, 160],  # N, W, S, E (Western Pacific)
    },
    'data/era5/2021.nc')
```

---

## Priority 2: Code Fixes

### Issue 3: Inference.py Incompatible with Model

**Problem:** `inference.py` tries to load `PhysicsInformedDiffusion` which doesn't exist, and uses methods that aren't implemented

**Status:** Partially fixed (import corrected to `PhysicsInformedDiffusionModel`)

**Remaining Issue:** The model doesn't have a `.predict()` method with DDIM sampling

**Fix Options:**

**Option A: Add DDIM Sampler to Model (RECOMMENDED)**

Create `models/diffusion/ddim_sampler.py`:

```python
"""DDIM Sampler for Fast Inference"""

import torch
import numpy as np

class DDIMSampler:
    """DDIM sampling for faster inference than DDPM"""
    
    def __init__(self, model, schedule, num_inference_steps=50):
        self.model = model
        self.schedule = schedule
        self.num_inference_steps = num_inference_steps
        
        # Create inference timestep schedule
        self.inference_timesteps = torch.linspace(
            0, schedule.num_timesteps - 1, 
            num_inference_steps
        ).long()
    
    @torch.no_grad()
    def sample(self, condition_dict, output_frames=8):
        """
        Generate samples using DDIM
        
        Args:
            condition_dict: {
                'past_latents': (B, T_past, C, H, W),
                'past_track': (B, T_past, 2),
                'past_intensity': (B, T_past)
            }
            output_frames: Number of frames to predict
        
        Returns:
            predictions: {
                'future_latents': (B, T_future, C, H, W),
                'track': (B, T_future, 2),
                'intensity': (B, T_future)
            }
        """
        device = condition_dict['past_latents'].device
        B, _, C, H, W = condition_dict['past_latents'].shape
        
        # Start from random noise
        z_t = torch.randn(B, output_frames, C, H, W, device=device)
        
        # Reverse diffusion process
        for i, t in enumerate(reversed(self.inference_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predictions = self.model(z_t, t_batch, condition_dict)
            noise_pred = predictions['noise']
            
            # DDIM update step
            alpha_t = self.schedule.alphas_bar[t]
            
            if i < len(self.inference_timesteps) - 1:
                alpha_prev = self.schedule.alphas_bar[self.inference_timesteps[-(i+2)]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)
            
            # Predict x0
            x0_pred = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Compute z_{t-1}
            z_t = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
        
        return {
            'future_latents': z_t,
            'track': predictions['track'],
            'intensity': predictions['intensity']
        }
```

Then update `PhysicsInformedDiffusionModel` to include:

```python
def predict(self, past_latents, past_track, past_intensity, 
            num_future_steps=8, ddim_steps=50):
    """Convenience method for inference"""
    from .ddim_sampler import DDIMSampler
    
    sampler = DDIMSampler(self, self.schedule, ddim_steps)
    
    condition_dict = {
        'past_latents': past_latents,
        'past_track': past_track,
        'past_intensity': past_intensity
    }
    
    return sampler.sample(condition_dict, num_future_steps)
```

**Option B: Fix inference.py to use model directly**

Update `inference.py` to implement DDIM sampling locally instead of relying on model method.

---

### Issue 4: Channel Count Mismatch Potential

**Problem:** Config says 48 channels, but with IBTrACS concatenation it becomes 52

**Check:**
```python
# In train_autoencoder.py or similar
print(f"Autoencoder in_channels: {model.in_channels}")
print(f"Dataset sample shape: {sample['past_frames'].shape}")
```

**Fix:**
- Either update config to 52 if using concat_ibtracs
- Or ensure concat_ibtracs=False in training

---

## Priority 3: Training Setup

### Issue 5: Running on CPU

**Problem:** Training on CPU is extremely slow

**Solutions:**

1. **Local GPU:** Check if you have CUDA:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

2. **Google Colab:**
- Upload code to Google Drive
- Use Colab with GPU (free T4 or Pro V100/A100)
- Mount Drive and run training

3. **University Cluster:**
- Most universities have GPU clusters
- Submit SLURM jobs

4. **Cloud Providers:**
- AWS: p3.2xlarge (V100)
- GCP: n1-standard-4 with V100
- Lambda Labs: cheapest GPU cloud

---

## Quick Start After Fixes

Once you've fixed the above:

```bash
# 1. Verify data is clean
python diagnose_data_issues.py

# 2. If NaN found, recompute stats
# (script will prompt you)

# 3. Retrain autoencoder with attention
# Edit configs/autoencoder_config.yaml:
#   use_attention: true
python train_autoencoder.py --config configs/autoencoder_config.yaml

# 4. Train diffusion model
python train_diffusion.py \
    --config configs/diffusion_config.yaml \
    --autoencoder checkpoints/autoencoder/best.pth

# 5. Evaluate
python evaluate.py \
    --autoencoder checkpoints/autoencoder/best.pth \
    --diffusion checkpoints/diffusion/best.pth \
    --data data/processed \
    --output results/
```

---

## Testing Checklist

Before serious training:

- [ ] Data has no NaN values
- [ ] Statistics file is valid (no NaN)
- [ ] At least 50% samples use real ERA5 (not synthetic)
- [ ] Channel counts match between dataset and models
- [ ] Can load a sample and pass through autoencoder
- [ ] Can load a sample and pass through diffusion model
- [ ] Inference script works end-to-end
- [ ] Have GPU access for training
- [ ] TensorBoard/W&B logging works

---

## Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Fix NaN issues | 1-2 days | ⚠️ TODO |
| Get ERA5 access | 2-3 days | ⚠️ TODO |
| Download ERA5 data | 3-5 days | ⚠️ TODO |
| Regenerate samples | 1 day | ⚠️ TODO |
| Fix inference code | 1 day | 🔄 IN PROGRESS |
| Test end-to-end | 1 day | ⏳ WAITING |
| Train autoencoder | 1 day | ✅ DONE (needs redo) |
| Train diffusion | 2-3 days | ⏳ WAITING |
| Run experiments | 1-2 weeks | ⏳ WAITING |
| Write paper | 2-3 weeks | ⏳ WAITING |

---

## Get Help

If stuck on any of these:

1. **Data issues:** Check `diagnose_data_issues.py` output
2. **ERA5 download:** See ERA5_SETUP.md
3. **Model errors:** Check model tests in each file
4. **Training issues:** Check logs in `logs/`

**Your pipeline is 70% done. These fixes will get you to 100%!**

