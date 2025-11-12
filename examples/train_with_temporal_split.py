"""
Example training script using temporal split dataset

This script demonstrates how to train a typhoon prediction model
using the correctly split dataset (by year) to avoid data leakage.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import logging

from data.datasets.typhoon_dataset import TyphoonDataset
from models.simple_baseline import SimpleBaselineModel
from trainers.trainer import TyphoonTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    
    print("="*80)
    print("TRAINING WITH TEMPORAL SPLIT (NO DATA LEAKAGE)")
    print("="*80)
    
    # Configuration
    config = {
        # Data configuration
        'data_dir': 'data/processed_temporal_split',  # Use temporal split dataset
        'use_temporal_split': True,  # IMPORTANT: Use temporal split
        'normalize': True,
        'concat_ibtracs': False,
        
        # Model configuration
        'in_channels': 48,  # ERA5 channels
        'hidden_dim': 128,
        'num_layers': 4,
        
        # Training configuration
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Validation
        'validate_every': 5,  # Validate every 5 epochs
        'save_best_only': True,
        
        # Output
        'output_dir': 'experiments/temporal_split_baseline',
        'save_checkpoint_every': 10
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create datasets with temporal split
    logger.info("\nCreating datasets...")
    
    try:
        train_dataset = TyphoonDataset(
            data_dir=config['data_dir'],
            split='train',
            normalize=config['normalize'],
            concat_ibtracs=config['concat_ibtracs'],
            use_temporal_split=True  # ← KEY: Use temporal split
        )
        
        val_dataset = TyphoonDataset(
            data_dir=config['data_dir'],
            split='val',
            normalize=config['normalize'],
            concat_ibtracs=config['concat_ibtracs'],
            use_temporal_split=True  # ← KEY: Use temporal split
        )
        
        test_dataset = TyphoonDataset(
            data_dir=config['data_dir'],
            split='test',
            normalize=config['normalize'],
            concat_ibtracs=config['concat_ibtracs'],
            use_temporal_split=True  # ← KEY: Use temporal split
        )
        
        logger.info(f"✅ Train dataset: {len(train_dataset)} samples")
        logger.info(f"✅ Val dataset: {len(val_dataset)} samples")
        logger.info(f"✅ Test dataset: {len(test_dataset)} samples")
        
    except ValueError as e:
        logger.error(f"\n❌ Error loading dataset: {e}")
        logger.error("\nPlease generate the temporal split dataset first:")
        logger.error("  cd data")
        logger.error("  python generate_data_by_year.py")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    logger.info(f"\n✅ Created dataloaders")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = SimpleBaselineModel(
        in_channels=config['in_channels'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    model = model.to(config['device'])
    
    logger.info(f"✅ Model created")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Create trainer
    trainer = TyphoonTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device'],
        output_dir=config['output_dir']
    )
    
    # Train
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    logger.info("\n⚠️  IMPORTANT: This is training with temporal split!")
    logger.info("   • Training years: 2021-2022")
    logger.info("   • Validation years: 2023")
    logger.info("   • Test years: 2024")
    logger.info("   • No data leakage!")
    logger.info("\n")
    
    try:
        trainer.train(
            num_epochs=config['num_epochs'],
            validate_every=config['validate_every'],
            save_checkpoint_every=config['save_checkpoint_every']
        )
        
        logger.info("\n✅ Training completed!")
        
        # Evaluate on test set
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*80)
        
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("\nTest Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\n✅ All done!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')
        logger.info("✅ Checkpoint saved")


if __name__ == "__main__":
    main()

