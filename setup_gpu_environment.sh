#!/bin/bash
# Setup script for GPU-enabled PyTorch environment

# Create new environment with Python 3.11.9
conda create -n pytorch_gpu python=3.11 -y

# Activate the environment
conda activate pytorch_gpu

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Reinstall your other dependencies
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: conda activate pytorch_gpu"

