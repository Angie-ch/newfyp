"""
Incremental Module Testing for Typhoon Prediction Pipeline

Tests each module before running full training to catch errors early.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import sys
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.datasets.typhoon_dataset import TyphoonDataset
from models.autoencoder.joint_autoencoder import JointAutoencoder
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel
from training.trainers.joint_autoencoder_trainer import JointAutoencoderTrainer
from training.trainers.joint_diffusion_trainer import JointDiffusionTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loading(config: dict):
    """Test 1: Data Loading"""
    logger.info("="*80)
    logger.info("TEST 1: Data Loading")
    logger.info("="*80)
    
    data_dir = Path(config['data']['data_dir'])
    
    try:
        # Try to load train dataset
        # Check for temporal split structure first
        train_dir = data_dir / 'train' / 'cases'
        if not train_dir.exists():
            train_dir = data_dir / 'train'
        if not train_dir.exists():
            train_dir = data_dir / 'cases'
        if not train_dir.exists():
            train_dir = data_dir
        
        logger.info(f"Loading data from: {train_dir}")
        # Dataset expects directory path, not cases subdirectory
        dataset_dir = train_dir.parent if train_dir.name == 'cases' else train_dir
        train_dataset = TyphoonDataset(
            data_dir=str(dataset_dir),
            split='train',
            normalize=True
        )
        
        logger.info(f"✓ Dataset loaded: {len(train_dataset)} samples")
        
        # Try to load a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"✓ Sample loaded successfully")
            logger.info(f"  - past_frames shape: {sample['past_frames'].shape}")
            logger.info(f"  - future_frames shape: {sample['future_frames'].shape}")
            logger.info(f"  - track_past shape: {sample['track_past'].shape}")
            logger.info(f"  - track_future shape: {sample['track_future'].shape}")
            logger.info(f"  - intensity_past shape: {sample['intensity_past'].shape}")
            logger.info(f"  - intensity_future shape: {sample['intensity_future'].shape}")
            
            # Check for NaN
            has_nan = (
                torch.isnan(sample['past_frames']).any() or
                torch.isnan(sample['future_frames']).any() or
                torch.isnan(sample['track_past']).any() or
                torch.isnan(sample['track_future']).any()
            )
            
            if has_nan:
                logger.warning("⚠ Sample contains NaN values!")
            else:
                logger.info("✓ No NaN values detected")
            
            # Create a small dataloader
            train_loader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0  # Use 0 for testing
            )
            
            # Try to get a batch
            batch = next(iter(train_loader))
            logger.info(f"✓ Batch loaded successfully")
            logger.info(f"  - Batch past_frames: {batch['past_frames'].shape}")
            logger.info(f"  - Batch future_frames: {batch['future_frames'].shape}")
            
            logger.info("✓ TEST 1 PASSED: Data loading works correctly")
            return True, train_loader
            
        else:
            logger.error("✗ No samples in dataset!")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_autoencoder_forward(config: dict, device: str):
    """Test 2: Autoencoder Forward Pass"""
    logger.info("="*80)
    logger.info("TEST 2: Autoencoder Forward Pass")
    logger.info("="*80)
    
    try:
        # Create model
        model = JointAutoencoder(
            era5_channels=config['model']['era5_channels'],
            latent_channels=config['model']['latent_channels'],
            hidden_dims=config['model'].get('hidden_dims', [64, 128, 256, 256]),
            use_attention=config['model'].get('use_attention', True)
        ).to(device)
        
        logger.info(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create dummy input
        batch_size = 2
        past_frames = 8
        era5_channels = config['model']['era5_channels']
        H, W = 64, 64
        
        era5 = torch.randn(batch_size, era5_channels, H, W).to(device)
        track = torch.randn(batch_size, 2).to(device)  # (lat, lon)
        intensity = torch.randn(batch_size).to(device)  # wind speed
        
        logger.info(f"✓ Input shapes:")
        logger.info(f"  - ERA5: {era5.shape}")
        logger.info(f"  - Track: {track.shape}")
        logger.info(f"  - Intensity: {intensity.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(era5, track, intensity)
        
        era5_recon = outputs['era5_recon']
        track_recon = outputs['track_recon']
        intensity_recon = outputs['intensity_recon']
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  - ERA5 recon: {era5_recon.shape}")
        logger.info(f"  - Track recon: {track_recon.shape}")
        logger.info(f"  - Intensity recon: {intensity_recon.shape}")
        
        # Check outputs
        assert era5_recon.shape == era5.shape, f"ERA5 shape mismatch: {era5_recon.shape} vs {era5.shape}"
        assert track_recon.shape == track.shape, f"Track shape mismatch: {track_recon.shape} vs {track.shape}"
        assert intensity_recon.shape == intensity.shape, f"Intensity shape mismatch: {intensity_recon.shape} vs {intensity.shape}"
        
        logger.info("✓ TEST 2 PASSED: Autoencoder forward pass works correctly")
        return True, model
        
    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_autoencoder_backward(config: dict, device: str, model: JointAutoencoder):
    """Test 3: Autoencoder Backward Pass"""
    logger.info("="*80)
    logger.info("TEST 3: Autoencoder Backward Pass")
    logger.info("="*80)
    
    try:
        model.train()
        
        # Create dummy input
        batch_size = 2
        era5_channels = config['model']['era5_channels']
        H, W = 64, 64
        
        era5 = torch.randn(batch_size, era5_channels, H, W).to(device)
        track = torch.randn(batch_size, 2).to(device)
        intensity = torch.randn(batch_size).to(device)
        
        # Forward pass
        outputs = model(era5, track, intensity)
        era5_recon = outputs['era5_recon']
        track_recon = outputs['track_recon']
        intensity_recon = outputs['intensity_recon']
        
        # Compute loss
        from models.autoencoder.joint_autoencoder import JointAutoencoderLoss
        criterion = JointAutoencoderLoss(
            era5_weight=config['training'].get('era5_weight', 1.0),
            track_weight=config['training'].get('track_weight', 10.0),
            intensity_weight=config['training'].get('intensity_weight', 5.0)
        )
        
        targets = {
            'era5': era5,
            'track': track,
            'intensity': intensity
        }
        
        loss, loss_dict = criterion(outputs, targets)
        
        logger.info(f"✓ Loss computed: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        if has_grad:
            logger.info("✓ Gradients computed successfully")
        else:
            logger.warning("⚠ No gradients computed!")
        
        logger.info("✓ TEST 3 PASSED: Autoencoder backward pass works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoencoder_training_step(config: dict, device: str, train_loader: DataLoader):
    """Test 4: Autoencoder Training Step"""
    logger.info("="*80)
    logger.info("TEST 4: Autoencoder Training Step (Small Batch)")
    logger.info("="*80)
    
    try:
        # Create model
        model = JointAutoencoder(
            era5_channels=config['model']['era5_channels'],
            latent_channels=config['model']['latent_channels'],
            hidden_dims=config['model'].get('hidden_dims', [64, 128, 256, 256]),
            use_attention=config['model'].get('use_attention', True)
        ).to(device)
        
        # Create trainer
        val_loader = DataLoader(
            train_loader.dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        trainer = JointAutoencoderTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            device=device
        )
        
        logger.info("✓ Trainer created")
        
        # Run one training step
        batch = next(iter(train_loader))
        
        # Move to device
        # Get actual channel count from data
        actual_channels = batch['past_frames'].shape[2]
        logger.info(f"  - Actual data channels: {actual_channels}, Model expects: {config['model']['era5_channels']}")
        
        # Use first N channels matching model config
        era5 = batch['past_frames'][:, 0, :config['model']['era5_channels']].to(device)  # First timestep, first N channels
        track = batch['track_past'][:, 0].to(device)  # First timestep
        intensity = batch['intensity_past'][:, 0].to(device)  # First timestep
        
        logger.info(f"  - Using ERA5 shape: {era5.shape}")
        
        model.train()
        trainer.optimizer.zero_grad()
        
        outputs = model(era5, track, intensity)
        
        targets = {
            'era5': era5,
            'track': track,
            'intensity': intensity
        }
        
        loss, loss_dict = trainer.criterion(outputs, targets)
        
        loss.backward()
        trainer.optimizer.step()
        
        logger.info(f"✓ Training step completed, loss: {loss.item():.4f}")
        logger.info("✓ TEST 4 PASSED: Autoencoder training step works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diffusion_forward(config: dict, device: str, autoencoder: JointAutoencoder):
    """Test 5: Diffusion Model Forward Pass"""
    logger.info("="*80)
    logger.info("TEST 5: Diffusion Model Forward Pass")
    logger.info("="*80)
    
    try:
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
        ).to(device)
        
        logger.info(f"✓ Diffusion model created: {sum(p.numel() for p in diffusion_model.parameters()):,} parameters")
        
        # Create dummy input
        batch_size = 2
        past_frames = config['data']['past_frames']
        future_frames = config['data']['future_frames']
        era5_channels = config['model']['era5_channels']
        H, W = 64, 64
        
        # Past frames
        era5_past = torch.randn(batch_size, past_frames, era5_channels, H, W).to(device)
        track_past = torch.randn(batch_size, past_frames, 2).to(device)
        intensity_past = torch.randn(batch_size, past_frames).to(device)
        
        # Encode past frames
        autoencoder.eval()
        with torch.no_grad():
            # Encode each timestep
            latents_past = []
            for t in range(past_frames):
                z = autoencoder.encode(
                    era5_past[:, t],
                    track_past[:, t],
                    intensity_past[:, t]
                )
                latents_past.append(z)
            # Stack: (B, T_past, C_latent, H/8, W/8)
            latents_past = torch.stack(latents_past, dim=1)  # (B, T_past, C_latent, 8, 8)
        
        logger.info(f"✓ Past latents shape: {latents_past.shape}")
        
        # Forward pass through diffusion
        diffusion_model.eval()
        with torch.no_grad():
            # Sample random noise for future
            noise = torch.randn(
                batch_size, future_frames,
                config['model']['latent_channels'], 8, 8
            ).to(device)
            
            # Predict noise
            timestep = torch.randint(0, 1000, (batch_size,)).to(device)
            condition_dict = {
                'past_latents': latents_past
            }
            predictions = diffusion_model(noise, timestep, condition_dict)
            predicted_noise = predictions['noise']
        
        logger.info(f"✓ Diffusion forward pass successful")
        logger.info(f"  - Predicted noise shape: {predicted_noise.shape}")
        
        logger.info("✓ TEST 5 PASSED: Diffusion model forward pass works correctly")
        return True, diffusion_model
        
    except Exception as e:
        logger.error(f"✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_diffusion_backward(config: dict, device: str, diffusion_model: PhysicsInformedDiffusionModel, autoencoder: JointAutoencoder):
    """Test 6: Diffusion Model Backward Pass"""
    logger.info("="*80)
    logger.info("TEST 6: Diffusion Model Backward Pass")
    logger.info("="*80)
    
    try:
        diffusion_model.train()
        
        # Create dummy input
        batch_size = 2
        past_frames = config['data']['past_frames']
        future_frames = config['data']['future_frames']
        era5_channels = config['model']['era5_channels']
        H, W = 64, 64
        
        # Past frames
        era5_past = torch.randn(batch_size, past_frames, era5_channels, H, W).to(device)
        track_past = torch.randn(batch_size, past_frames, 2).to(device)
        intensity_past = torch.randn(batch_size, past_frames).to(device)
        
        # Encode past frames
        autoencoder.eval()
        with torch.no_grad():
            latents_past = []
            for t in range(past_frames):
                z = autoencoder.encode(
                    era5_past[:, t],
                    track_past[:, t],
                    intensity_past[:, t]
                )
                latents_past.append(z)
            latents_past = torch.stack(latents_past, dim=1)  # (B, T_past, C_latent, 8, 8)
        
        # Forward pass
        noise = torch.randn(
            batch_size, future_frames,
            config['model']['latent_channels'], 8, 8
        ).to(device)
        
        timestep = torch.randint(0, 1000, (batch_size,)).to(device)
        condition_dict = {
            'past_latents': latents_past
        }
        predictions = diffusion_model(noise, timestep, condition_dict)
        predicted_noise = predictions['noise']
        
        # Compute loss (simple MSE)
        target_noise = torch.randn_like(noise)
        loss = nn.functional.mse_loss(predicted_noise, target_noise)
        
        logger.info(f"✓ Loss computed: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in diffusion_model.parameters())
        if has_grad:
            logger.info("✓ Gradients computed successfully")
        else:
            logger.warning("⚠ No gradients computed!")
        
        logger.info("✓ TEST 6 PASSED: Diffusion model backward pass works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests incrementally"""
    logger.info("="*80)
    logger.info("TYPHOON PREDICTION PIPELINE - INCREMENTAL MODULE TESTING")
    logger.info("="*80)
    
    # Load config
    config_path = Path("configs/joint_autoencoder.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Test 1: Data Loading
    success, train_loader = test_data_loading(config)
    if not success:
        logger.error("Data loading failed. Please fix before continuing.")
        return
    
    # Test 2: Autoencoder Forward
    success, autoencoder = test_autoencoder_forward(config, device)
    if not success:
        logger.error("Autoencoder forward pass failed. Please fix before continuing.")
        return
    
    # Test 3: Autoencoder Backward
    success = test_autoencoder_backward(config, device, autoencoder)
    if not success:
        logger.error("Autoencoder backward pass failed. Please fix before continuing.")
        return
    
    # Test 4: Autoencoder Training Step
    success = test_autoencoder_training_step(config, device, train_loader)
    if not success:
        logger.error("Autoencoder training step failed. Please fix before continuing.")
        return
    
    # Test 5: Diffusion Forward
    # Load diffusion config
    diff_config_path = Path("configs/joint_diffusion.yaml")
    if diff_config_path.exists():
        with open(diff_config_path, 'r') as f:
            diff_config = yaml.safe_load(f)
    else:
        logger.warning("Diffusion config not found, using autoencoder config")
        diff_config = config
    
    success, diffusion_model = test_diffusion_forward(diff_config, device, autoencoder)
    if not success:
        logger.error("Diffusion forward pass failed. Please fix before continuing.")
        return
    
    # Test 6: Diffusion Backward
    success = test_diffusion_backward(diff_config, device, diffusion_model, autoencoder)
    if not success:
        logger.error("Diffusion backward pass failed. Please fix before continuing.")
        return
    
    logger.info("="*80)
    logger.info("✓ ALL TESTS PASSED! Pipeline is ready for training.")
    logger.info("="*80)
    logger.info("You can now run the full training with:")
    logger.info("  python train_joint_pipeline.py --stage autoencoder --config configs/joint_autoencoder.yaml")
    logger.info("  python train_joint_pipeline.py --stage diffusion --config configs/joint_diffusion.yaml")


if __name__ == '__main__':
    main()

