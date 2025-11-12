"""
Training Script for Physics-Informed Diffusion Model

Usage:
    python train_diffusion.py --config configs/diffusion_config.yaml --autoencoder checkpoints/autoencoder/best.pth
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.diffusion import PhysicsInformedDiffusionModel
from models.autoencoder import SpatialAutoencoder
from data.datasets import TyphoonDataset
from training.trainers import DiffusionTrainer


def load_autoencoder(path: str, device: str) -> SpatialAutoencoder:
    """Load pretrained autoencoder"""
    print(f"Loading autoencoder from {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Infer model architecture from state_dict
    state_dict = checkpoint['model_state_dict']
    in_channels = state_dict['encoder_input.weight'].shape[1]
    
    # Check if attention layers exist
    has_attention = any('qkv' in key for key in state_dict.keys())
    
    print(f"  Inferred model config:")
    print(f"    in_channels: {in_channels}")
    print(f"    use_attention: {has_attention}")
    
    model = SpatialAutoencoder(
        in_channels=in_channels,
        latent_channels=8,
        hidden_dims=[64, 128, 256, 256],
        use_attention=has_attention
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Autoencoder loaded successfully")
    
    return model


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
    
    # Load pretrained autoencoder
    autoencoder = load_autoencoder(args.autoencoder, device)
    
    # Create diffusion model
    model = PhysicsInformedDiffusionModel(
        latent_channels=config['model']['latent_channels'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        depth=config['model'].get('depth', 3),
        output_frames=config['data'].get('output_frames', 8),
        use_physics_projector=True,
        use_spiral_attention=config['model'].get('use_spiral_attention', True),
        use_multiscale_temporal=config['model'].get('use_multiscale_temporal', True)
    )
    
    print(f"\nDiffusion model created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = TyphoonDataset(
        data_dir=str(Path(config['data']['data_dir']) / 'train'),
        split='train',
        normalize=False,
        concat_ibtracs=False
    )
    
    val_dataset = TyphoonDataset(
        data_dir=str(Path(config['data']['data_dir']) / 'val'),
        split='val',
        normalize=False,
        concat_ibtracs=False
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
    
    # Update config with full settings
    training_config = config['training'].copy()
    training_config.update({
        'timesteps': config['diffusion']['timesteps'],
        'beta_start': config['diffusion']['beta_start'],
        'beta_end': config['diffusion']['beta_end'],
        'beta_schedule': config['diffusion']['beta_schedule']
    })
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
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
    parser = argparse.ArgumentParser(description='Train Physics-Informed Diffusion Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--autoencoder', type=str, required=True,
                        help='Path to pretrained autoencoder checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    
    args = parser.parse_args()
    
    main(args)

