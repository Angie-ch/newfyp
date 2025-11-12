# Physics-Informed Multi-Task Typhoon Prediction Pipeline

## Overview

A state-of-the-art deep learning system for typhoon forecasting that combines:
- **Physics-Informed Diffusion**: Constrained predictions obeying atmospheric physics laws
- **Typhoon-Aware Architecture**: Custom spiral attention and multi-scale temporal modeling
- **Multi-Task Learning**: Simultaneous prediction of structure, track, and intensity

## Architecture

```
Input: 12 frames (ERA5 + IBTrACS) → Output: 8 future frames + track + intensity
```

### Key Components

1. **Spatial Autoencoder**: Compresses frames by 8x (256×256 → 32×32 latent)
2. **Physics-Informed Diffusion Model**: 
   - Spiral attention for cyclone dynamics
   - Multi-scale temporal convolutions
   - Physics constraint layers
3. **Multi-Task Heads**: Structure + Track + Intensity predictions

## Project Structure

```
typhoon_prediction/
├── configs/                # Configuration files
├── data/                   # Data processing
│   ├── preprocessing/      # ERA5 & IBTrACS preprocessing
│   └── datasets/          # PyTorch datasets
├── models/                 # Model architectures
│   ├── autoencoder/       # Spatial compression
│   ├── diffusion/         # Diffusion models
│   └── components/        # Reusable components
├── training/              # Training logic
│   ├── losses/           # Loss functions
│   └── trainers/         # Training loops
├── evaluation/            # Evaluation & metrics
│   ├── metrics/          # Performance metrics
│   └── visualizations/   # Plotting tools
└── utils/                # Utility functions

```

## Installation

```bash
# Create conda environment
conda create -n typhoon python=3.10
conda activate typhoon

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preprocessing

```python
from data.preprocessing.era5_processor import ERA5Processor
from data.preprocessing.ibtracs_processor import IBTrACSProcessor

# Process ERA5 data
era5_processor = ERA5Processor(data_dir='path/to/era5')
era5_processor.process_all()

# Process IBTrACS data
ibtracs_processor = IBTrACSProcessor(data_dir='path/to/ibtracs')
ibtracs_processor.process_all()
```

### 2. Train Autoencoder

```bash
python train_autoencoder.py --config configs/autoencoder_config.yaml
```

### 3. Train Diffusion Model

```bash
python train_diffusion.py --config configs/diffusion_config.yaml
```

### 4. Inference

```python
from inference import TyphoonPredictor

predictor = TyphoonPredictor(
    autoencoder_path='checkpoints/autoencoder.pth',
    diffusion_path='checkpoints/diffusion.pth'
)

predictions = predictor.predict(past_frames, past_track, past_intensity)
# Returns: {'frames': ..., 'track': ..., 'intensity': ...}
```

## Key Features

### Innovation #1: Physics-Informed Diffusion
- Geostrophic balance constraints
- Mass conservation laws
- Wind-pressure relationship
- Temporal smoothness

### Innovation #2: Typhoon-Aware Architecture
- Spiral attention mechanism biased toward cyclone patterns
- Multi-scale temporal modeling (3/5/9 frame windows)
- Domain-specific inductive biases

### Innovation #3: Multi-Task Learning
- Joint prediction improves all tasks
- Shared representations
- Cross-task consistency losses

## Performance

Expected improvements over baselines:
- **Track Error (48h)**: ~120 km (30%+ improvement)
- **Intensity MAE**: ~5 m/s (35%+ improvement)
- **Physics Consistency**: 95% pass rate

## Citation

```bibtex
@article{typhoon2025,
  title={Physics-Informed Video Diffusion for Tropical Cyclone Prediction: A Multi-Task Learning Approach},
  author={Your Name},
  journal={TBD},
  year={2025}
}
```

## License

MIT License

