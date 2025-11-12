#!/bin/bash
# Start Training Pipeline for Typhoon Prediction

echo "=========================================="
echo "Typhoon Prediction Training Pipeline"
echo "=========================================="
echo ""

# Check if CUDA is available
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    DEVICE="cuda"
else
    DEVICE="cpu"
    echo "Warning: CUDA not available, using CPU (training will be slower)"
fi

echo ""
echo "Step 1: Testing all modules..."
python test_pipeline_modules.py

if [ $? -ne 0 ]; then
    echo "ERROR: Module tests failed. Please fix errors before training."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Training Joint Autoencoder"
echo "=========================================="
echo "This will train the autoencoder that encodes ERA5 + IBTrACS together"
echo ""

python train_joint_pipeline.py \
    --stage autoencoder \
    --config configs/joint_autoencoder.yaml \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "ERROR: Autoencoder training failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Training Diffusion Model"
echo "=========================================="
echo "This will train the diffusion model on the latent space"
echo ""

python train_joint_pipeline.py \
    --stage diffusion \
    --config configs/joint_diffusion.yaml \
    --device $DEVICE

if [ $? -ne 0 ]; then
    echo "ERROR: Diffusion training failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved in:"
echo "  - checkpoints/joint_autoencoder/"
echo "  - checkpoints/joint_diffusion/"
echo ""
echo "Logs saved in:"
echo "  - logs/joint_autoencoder/"
echo "  - logs/joint_diffusion/"

