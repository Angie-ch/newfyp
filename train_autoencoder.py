"""
Training Script for Spatial Autoencoder

Usage:
    python train_autoencoder.py --config configs/autoencoder_config.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.autoencoder import SpatialAutoencoder
from data.datasets import TyphoonDataset
from training.trainers import AutoencoderTrainer


def main(args):
    """Main training function"""
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from {args.config}")
    print(yaml.dump(config, default_flow_style=False))
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create model
    model = SpatialAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_channels=config['model']['latent_channels'],
        hidden_dims=config['model']['hidden_dims'],
        use_attention=config['model'].get('use_attention', True)
    )
    
    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = TyphoonDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        transform=DataAugmentation() if args.augment else None
    )
    
    val_dataset = TyphoonDataset(
        data_dir=config['data']['data_dir'],
        split='val'
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )
    
    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    print("\nStarting training...")
    trainer.train(epochs=config['training']['epochs'])
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Spatial Autoencoder')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    
    args = parser.parse_args()
    
    main(args)

