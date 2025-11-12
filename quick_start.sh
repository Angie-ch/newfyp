#!/bin/bash

# Quick Start Script for Typhoon Prediction Pipeline
# This script provides example commands for the complete workflow

set -e  # Exit on error

echo "=================================="
echo "Typhoon Prediction Quick Start"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Data Preprocessing
echo -e "\n${YELLOW}Step 1: Data Preprocessing${NC}"
echo "Command:"
echo "  python preprocess_data.py \\"
echo "    --era5_dir data/raw/era5 \\"
echo "    --ibtracs data/raw/IBTrACS.WP.v04r00.csv \\"
echo "    --output data/processed \\"
echo "    --start_date 2015-01-01 \\"
echo "    --end_date 2020-12-31 \\"
echo "    --compute_stats"
echo ""
read -p "Run preprocessing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python preprocess_data.py \
        --era5_dir data/raw/era5 \
        --ibtracs data/raw/IBTrACS.WP.v04r00.csv \
        --output data/processed \
        --start_date 2015-01-01 \
        --end_date 2020-12-31 \
        --compute_stats
    echo -e "${GREEN}✓ Preprocessing complete${NC}"
fi

# Step 2: Train Autoencoder
echo -e "\n${YELLOW}Step 2: Train Autoencoder${NC}"
echo "Command:"
echo "  python train_autoencoder.py \\"
echo "    --config configs/autoencoder_config.yaml \\"
echo "    --augment"
echo ""
read -p "Train autoencoder? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python train_autoencoder.py \
        --config configs/autoencoder_config.yaml \
        --augment
    echo -e "${GREEN}✓ Autoencoder training complete${NC}"
fi

# Step 3: Train Diffusion Model
echo -e "\n${YELLOW}Step 3: Train Diffusion Model${NC}"
echo "Command:"
echo "  python train_diffusion.py \\"
echo "    --config configs/diffusion_config.yaml \\"
echo "    --autoencoder checkpoints/autoencoder/best.pth \\"
echo "    --augment"
echo ""
read -p "Train diffusion model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python train_diffusion.py \
        --config configs/diffusion_config.yaml \
        --autoencoder checkpoints/autoencoder/best.pth \
        --augment
    echo -e "${GREEN}✓ Diffusion model training complete${NC}"
fi

# Step 4: Evaluate
echo -e "\n${YELLOW}Step 4: Evaluate Model${NC}"
echo "Command:"
echo "  python evaluate.py \\"
echo "    --autoencoder checkpoints/autoencoder/best.pth \\"
echo "    --diffusion checkpoints/diffusion/best.pth \\"
echo "    --data data/processed \\"
echo "    --output results/evaluation.json"
echo ""
read -p "Run evaluation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python evaluate.py \
        --autoencoder checkpoints/autoencoder/best.pth \
        --diffusion checkpoints/diffusion/best.pth \
        --data data/processed \
        --output results/evaluation.json
    echo -e "${GREEN}✓ Evaluation complete${NC}"
fi

echo -e "\n${GREEN}=================================="
echo "Quick Start Complete!"
echo "==================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: checkpoints/"
echo "  - Logs: logs/"
echo "  - Evaluation: results/evaluation.json"
echo ""
echo "Monitor training:"
echo "  tensorboard --logdir logs/"
echo ""

