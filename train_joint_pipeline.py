"""
Complete Training Pipeline for Joint ERA5 + IBTrACS Model

This script trains the complete pipeline:
1. Joint Autoencoder: Encodes ERA5 + IBTrACS together, decodes separately
2. Diffusion Model: Operates on unified latent space
3. Trajectory Prediction: Uses decoded outputs for prediction

Usage:
    # Train joint autoencoder
    python train_joint_pipeline.py --stage autoencoder --config configs/joint_autoencoder.yaml
    
    # Train diffusion model
    python train_joint_pipeline.py --stage diffusion --config configs/joint_diffusion.yaml
    
    # Train both sequentially
    python train_joint_pipeline.py --stage both --config configs/joint_pipeline.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
import logging
import signal
import sys

from data.datasets.typhoon_dataset import TyphoonDataset
from torch.utils.data import DataLoader

from models.autoencoder.joint_autoencoder import JointAutoencoder
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel

from training.trainers.joint_autoencoder_trainer import JointAutoencoderTrainer
from training.trainers.joint_diffusion_trainer import JointDiffusionTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store dataloaders for cleanup
_active_dataloaders = []


def cleanup_dataloaders():
    """Clean up all active dataloaders to prevent semaphore leaks"""
    global _active_dataloaders
    for loader in _active_dataloaders:
        if hasattr(loader, '_iterator'):
            try:
                loader._iterator = None
            except:
                pass
    _active_dataloaders.clear()


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    logger.info("\n\nInterrupt received. Cleaning up...")
    cleanup_dataloaders()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, stage: str = 'autoencoder'):
    """Create train and validation dataloaders"""
    data_dir = Path(config['data']['data_dir'])
    
    # Training dataset
    # Pass root data_dir, dataset will handle split internally
    train_dataset = TyphoonDataset(
        data_dir=str(data_dir),
        split='train',
        normalize=True,
        concat_ibtracs=False  # Don't concatenate IBTrACS to ERA5 channels
    )
    
    # Validation dataset
    val_dataset = TyphoonDataset(
        data_dir=str(data_dir),
        split='val',
        normalize=True,
        concat_ibtracs=False  # Don't concatenate IBTrACS to ERA5 channels
    )
    
    # Get batch size based on stage
    if stage == 'autoencoder':
        # Check if nested structure exists (joint_pipeline.yaml) or flat (joint_autoencoder.yaml)
        if 'autoencoder' in config['training']:
            batch_size = config['training']['autoencoder']['batch_size']
        else:
            batch_size = config['training'].get('batch_size', 8)
    elif stage == 'diffusion':
        if 'diffusion' in config['training']:
            batch_size = config['training']['diffusion']['batch_size']
        else:
            batch_size = config['training'].get('batch_size', 4)
    else:
        # Fallback: try to get from training directly
        batch_size = config['training'].get('batch_size', 8)
    
    num_workers = config['training'].get('num_workers', 4)
    
    # On macOS, multiprocessing can cause issues - use spawn method if needed
    if num_workers > 0 and sys.platform == 'darwin':
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
    
    # Dataloaders
    # Note: persistent_workers only works when num_workers > 0
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
    }
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True  # Keep workers alive between epochs
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    # Register for cleanup
    global _active_dataloaders
    _active_dataloaders = [train_loader, val_loader]
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"DataLoader num_workers: {num_workers}")
    
    return train_loader, val_loader


def train_autoencoder(config: dict, device: str):
    """
    Train joint autoencoder
    
    Encodes ERA5 + IBTrACS together -> Unified Latent
    Decodes Unified Latent -> ERA5 + IBTrACS separately
    """
    logger.info("="*80)
    logger.info("STAGE 1: Training Joint Autoencoder")
    logger.info("="*80)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, stage='autoencoder')
    
    # Create model
    model = JointAutoencoder(
        era5_channels=config['model']['era5_channels'],
        latent_channels=config['model']['latent_channels'],
        hidden_dims=config['model'].get('hidden_dims', [64, 128, 256, 256]),
        use_attention=config['model'].get('use_attention', True),
        dropout=config['model'].get('dropout', 0.1)
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get training config for autoencoder
    # Check if nested structure exists (joint_pipeline.yaml) or flat (joint_autoencoder.yaml)
    if 'autoencoder' in config['training']:
        train_config = config['training']['autoencoder'].copy()
        train_config['log_interval'] = config['training'].get('log_interval', 50)
        train_config['save_interval'] = config['training'].get('save_interval', 5)
    else:
        # Flat structure: use training config directly
        train_config = config['training'].copy()
    
    # Create trainer
    trainer = JointAutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Train
    if 'autoencoder' in config['training']:
        epochs = config['training']['autoencoder']['epochs']
    else:
        epochs = config['training'].get('epochs', 50)
    trainer.train(epochs=epochs)
    
    logger.info("Joint autoencoder training complete!")
    
    return model


def train_diffusion(config: dict, device: str, autoencoder_path: str = None):
    """
    Train diffusion model with joint autoencoder
    
    Uses unified latent space from joint autoencoder
    """
    logger.info("="*80)
    logger.info("STAGE 2: Training Diffusion Model")
    logger.info("="*80)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, stage='diffusion')
    
    # Load pretrained joint autoencoder
    if autoencoder_path is None:
        autoencoder_path = config['model'].get('autoencoder_checkpoint')
        if autoencoder_path is None:
            raise ValueError("autoencoder_checkpoint must be provided in config or as argument")
    
    logger.info(f"Loading joint autoencoder from {autoencoder_path}")
    
    autoencoder = JointAutoencoder(
        era5_channels=config['model']['era5_channels'],
        latent_channels=config['model']['latent_channels'],
        hidden_dims=config['model'].get('hidden_dims', [64, 128, 256, 256]),
        use_attention=config['model'].get('use_attention', True),
        dropout=config['model'].get('dropout', 0.1)
    )
    
    checkpoint = torch.load(autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # Create diffusion model
    diffusion_model = PhysicsInformedDiffusionModel(
        latent_channels=config['model']['latent_channels'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        depth=config['model'].get('depth', 3),
        output_frames=config['data']['future_frames'],
        use_physics_projector=config['model'].get('use_physics', True),
        use_spiral_attention=config['model'].get('use_spiral_attention', True),
        use_multiscale_temporal=config['model'].get('use_multiscale_temporal', True)
    )
    
    logger.info(f"Diffusion model parameters: {sum(p.numel() for p in diffusion_model.parameters()):,}")
    
    # Get training config for diffusion
    train_config = config['training']['diffusion'].copy()
    train_config['log_interval'] = config['training'].get('log_interval', 50)
    train_config['save_interval'] = config['training'].get('save_interval', 5)
    
    # Create trainer
    trainer = JointDiffusionTrainer(
        model=diffusion_model,
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Train
    epochs = config['training']['diffusion']['epochs']
    trainer.train(epochs=epochs)
    
    logger.info("Diffusion model training complete!")
    
    return diffusion_model


def main():
    parser = argparse.ArgumentParser(description='Train Joint ERA5 + IBTrACS Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--stage', type=str, choices=['autoencoder', 'diffusion', 'both'],
                       default='both', help='Training stage')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--autoencoder_checkpoint', type=str, default=None,
                       help='Path to pretrained autoencoder (for diffusion stage)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device - respect user's choice, but validate CUDA availability
    # Map 'gpu' to 'cuda' for convenience
    if args.device.lower() == 'gpu':
        args.device = 'cuda'
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Train
    if args.stage == 'autoencoder':
        train_autoencoder(config, device)
    
    elif args.stage == 'diffusion':
        train_diffusion(config, device, args.autoencoder_checkpoint)
    
    elif args.stage == 'both':
        # Train autoencoder first
        autoencoder = train_autoencoder(config, device)
        
        # Get autoencoder checkpoint path
        if 'autoencoder' in config['training']:
            save_dir = config['training']['autoencoder']['save_dir']
        else:
            save_dir = config['training'].get('save_dir', 'checkpoints/joint_autoencoder')
        ae_checkpoint_path = Path(save_dir) / 'best.pth'
        
        # Train diffusion model
        train_diffusion(config, device, str(ae_checkpoint_path))
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()

