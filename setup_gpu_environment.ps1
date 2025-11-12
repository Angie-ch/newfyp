# PowerShell script for GPU-enabled PyTorch environment setup
# Run this in Anaconda Prompt or PowerShell with conda initialized

# Create new environment with Python 3.11
conda create -n pytorch_gpu python=3.11 -y

# Activate the environment
conda activate pytorch_gpu

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Reinstall your other dependencies
pip install -r requirements.txt

Write-Host "Setup complete! Activate the environment with: conda activate pytorch_gpu" -ForegroundColor Green

