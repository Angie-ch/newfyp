#!/usr/bin/env python3
"""
Complete Pipeline with Real IBTrACS WP + ERA5 Data
===================================================

This script runs the entire typhoon prediction pipeline using only real data:
- IBTrACS Western Pacific typhoon tracks (automatic download)
- ERA5 meteorological reanalysis (default, with synthetic fallback option)

Steps:
1. Load real typhoon data from IBTrACS WP
2. Generate training samples with ERA5 or synthetic frames
3. Train spatial autoencoder
4. Train physics-informed diffusion model
5. Evaluate on test set
6. Generate comprehensive visualizations

Usage:
    # Default: Uses ERA5 if cached, downloads if needed
    python run_real_data_pipeline.py --n-samples 100

    # Force download ERA5 data (slow but ensures latest data)
    python run_real_data_pipeline.py --n-samples 100 --download-era5
    
    # Use synthetic frames instead of ERA5
    python run_real_data_pipeline.py --n-samples 100 --no-use-era5
    
    # Quick test (minimal training)
    python run_real_data_pipeline.py --n-samples 20 --quick-test
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.real_data_loader import IBTrACSLoader
from data.datasets.typhoon_dataset import TyphoonDataset
from models.autoencoder.autoencoder import SpatialAutoencoder
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel
from training.trainers.autoencoder_trainer import AutoencoderTrainer
from training.trainers.diffusion_trainer import DiffusionTrainer
from evaluation.metrics.prediction_metrics import PredictionMetrics
from inference import TyphoonPredictor
from visualize_results import TyphoonVisualizer


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def print_progress(message, success=True):
    """Print progress message"""
    symbol = "✓" if success else "✗"
    print(f"{symbol} {message}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run complete pipeline with real IBTrACS WP + ERA5 data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of training samples to generate')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year for data')
    parser.add_argument('--end-year', type=int, default=2024,
                        help='End year for data')
    parser.add_argument('--use-era5', action='store_true', default=True,
                        help='Use ERA5 data (default: True, use --no-use-era5 to disable)')
    parser.add_argument('--no-use-era5', action='store_false', dest='use_era5',
                        help='Use synthetic data instead of ERA5')
    parser.add_argument('--download-era5', action='store_true',
                        help='Download ERA5 data if not cached (slow)')
    parser.add_argument('--max-storms-era5', type=int, default=10,
                        help='Maximum number of storms to download ERA5 data for (default: 10)')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory for raw data')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Directory for processed data')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--autoencoder-epochs', type=int, default=20,
                        help='Number of epochs for autoencoder')
    parser.add_argument('--diffusion-epochs', type=int, default=30,
                        help='Number of epochs for diffusion model')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with minimal training')
    
    # Model parameters
    parser.add_argument('--latent-channels', type=int, default=8,
                        help='Number of latent channels')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for diffusion model')
    
    # Evaluation parameters
    parser.add_argument('--num-inference-steps', type=int, default=50,
                        help='Number of inference steps for diffusion')
    parser.add_argument('--eval-samples', type=int, default=10,
                        help='Number of samples to evaluate and visualize')
    
    # Output parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--results-dir', type=str, default='results_real_data',
                        help='Directory for results')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    
    # Execution control
    parser.add_argument('--skip-data-generation', action='store_true',
                        help='Skip data generation (use existing)')
    parser.add_argument('--skip-autoencoder', action='store_true',
                        help='Skip autoencoder training (use existing)')
    parser.add_argument('--skip-diffusion', action='store_true',
                        help='Skip diffusion training (use existing)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories"""
    print_header("SETTING UP DIRECTORIES")
    
    dirs = [
        args.data_dir,
        args.output_dir,
        f"{args.output_dir}/cases",
        f"{args.checkpoint_dir}/autoencoder",
        f"{args.checkpoint_dir}/diffusion",
        f"{args.log_dir}/autoencoder",
        f"{args.log_dir}/diffusion",
        args.results_dir,
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print_progress(f"Created/verified directory: {d}")


def generate_real_data(args):
    """Generate training data from real IBTrACS WP + ERA5"""
    print_header("STEP 1: LOADING REAL TYPHOON DATA")
    
    print(f"Data source: IBTrACS Western Pacific")
    print(f"Time range: {args.start_year}-{args.end_year}")
    print(f"Target samples: {args.n_samples}")
    if args.use_era5:
        print(f"Meteorological data: ERA5 (48 channels)")
        if args.download_era5:
            print(f"⚠️  ERA5 download enabled - this may take a while!")
    else:
        print(f"Meteorological data: Synthetic (48 channels)")
    
    # Create loader
    loader = IBTrACSLoader(data_dir=args.data_dir)
    
    # Load IBTrACS data
    print("\nLoading IBTrACS Western Pacific data...")
    df = loader.load_ibtracs()
    print_progress(f"Loaded {len(df)} IBTrACS records")
    
    # Filter for strong typhoons in time range
    storm_ids = loader.filter_typhoons(
        df, 
        start_year=args.start_year, 
        end_year=args.end_year
    )
    print_progress(f"Found {len(storm_ids)} Western Pacific typhoons")
    
    if len(storm_ids) == 0:
        print("✗ No storms found in the specified time range!")
        print("  Try expanding the year range or check data availability")
        sys.exit(1)
    
    # Show sample storms
    print("\nSample storms:")
    for i, sid in enumerate(storm_ids[:5]):
        storm_data = df[df['SID'] == sid].iloc[0]
        name = storm_data['NAME'] if 'NAME' in storm_data else 'UNNAMED'
        print(f"  {i+1}. {name} ({sid})")
    if len(storm_ids) > 5:
        print(f"  ... and {len(storm_ids) - 5} more")
    
    # Generate training samples
    print(f"\nGenerating {args.n_samples} training samples...")
    samples = loader.create_dataset(
        n_samples=args.n_samples,
        start_year=args.start_year,
        end_year=args.end_year,
        use_era5=args.use_era5,
        download_era5=args.download_era5,
        max_storms_download=args.max_storms_era5,
        past_timesteps=8,  # 8 timesteps of historical data
        future_timesteps=12  # 12 timesteps to predict into future
    )
    
    print_progress(f"Generated {len(samples)} samples")
    
    # Save samples to disk
    print("\nSaving samples to disk...")
    metadata = []
    
    for i, sample in enumerate(samples):
        # Save as .npz file
        output_path = Path(args.output_dir) / 'cases' / f'case_{i:04d}.npz'
        np.savez_compressed(
            output_path,
            past_frames=sample['past_frames'],
            future_frames=sample['future_frames'],
            track_past=sample['past_track'],
            track_future=sample['future_track'],
            intensity_past=sample['past_intensity'],
            intensity_future=sample['future_intensity'],
            case_id=sample['storm_id']
        )
        
        # Store metadata
        metadata.append({
            'case_id': i,
            'storm_id': sample['storm_id'],
            'storm_name': sample.get('storm_name', 'UNKNOWN'),
            'max_past_intensity': float(sample['past_intensity'].max()),
            'max_future_intensity': float(sample['future_intensity'].max()),
        })
        
        if (i + 1) % 20 == 0:
            print_progress(f"Saved {i + 1}/{len(samples)} cases...")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(Path(args.output_dir) / 'metadata.csv', index=False)
    print_progress(f"Saved metadata to {args.output_dir}/metadata.csv")
    
    # Compute and save statistics
    print("\nComputing dataset statistics...")
    all_past = np.stack([s['past_frames'] for s in samples], axis=0)
    all_future = np.stack([s['future_frames'] for s in samples], axis=0)
    
    stats = {
        'mean': float(all_past.mean()),
        'std': float(all_past.std()),
        'min': float(all_past.min()),
        'max': float(all_past.max()),
        'n_samples': len(samples),
        'shape': {
            'past_frames': list(samples[0]['past_frames'].shape),
            'future_frames': list(samples[0]['future_frames'].shape),
        },
        'data_source': 'IBTrACS_WP',
        'meteorological_data': 'ERA5' if args.use_era5 else 'Synthetic',
        'year_range': f"{args.start_year}-{args.end_year}",
        'generation_date': datetime.now().isoformat()
    }
    
    with open(Path(args.output_dir) / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print_progress(f"Dataset statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Shape: {stats['shape']}")
    
    return len(samples)


def train_autoencoder(args):
    """Train spatial autoencoder"""
    print_header("STEP 2: TRAINING SPATIAL AUTOENCODER")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = TyphoonDataset(args.output_dir, split='train')
    val_dataset = TyphoonDataset(args.output_dir, split='val')
    
    print_progress(f"Train: {len(train_dataset)} samples")
    print_progress(f"Val: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for macOS compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nCreating autoencoder model...")
    # Get number of channels from a sample
    sample = train_dataset[0]
    n_channels = sample['past_frames'].shape[1]  # (T, C, H, W) - get C
    
    model = SpatialAutoencoder(
        in_channels=n_channels,
        latent_channels=args.latent_channels,
        hidden_dims=[64, 128, 256, 256],
        use_attention=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print_progress(f"Model created: {n_params/1e6:.2f}M parameters")
    print(f"  Input channels: {n_channels}")
    print(f"  Latent channels: {args.latent_channels}")
    
    # Training config
    epochs = 3 if args.quick_test else args.autoencoder_epochs
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'epochs': epochs,
        'checkpoint_dir': f'{args.checkpoint_dir}/autoencoder',
        'log_dir': f'{args.log_dir}/autoencoder',
        'save_freq': max(1, epochs // 5),
        'in_channels': n_channels,
        'latent_channels': args.latent_channels,
        'hidden_dims': [64, 128, 256, 256],
        'use_attention': True
    }
    
    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    print(f"\nTraining autoencoder for {epochs} epochs...")
    trainer.train(epochs=epochs)
    
    print_progress(f"Training complete! Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {config['checkpoint_dir']}/best.pth")
    
    return model


def train_diffusion(args, autoencoder):
    """Train physics-informed diffusion model"""
    print_header("STEP 3: TRAINING PHYSICS-INFORMED DIFFUSION MODEL")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TyphoonDataset(args.output_dir, split='train')
    val_dataset = TyphoonDataset(args.output_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Ensure autoencoder is on correct device
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Create diffusion model
    print("\nCreating diffusion model...")
    model = PhysicsInformedDiffusionModel(
        latent_channels=args.latent_channels,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        depth=3,
        output_frames=8,
        use_physics_projector=True,
        use_spiral_attention=True,
        use_multiscale_temporal=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print_progress(f"Model created: {n_params/1e6:.2f}M parameters")
    
    # Training config
    epochs = 3 if args.quick_test else args.diffusion_epochs
    config = {
        'learning_rate': args.learning_rate * 2,  # Slightly higher for diffusion
        'weight_decay': 0.01,
        'epochs': epochs,
        'timesteps': 100 if args.quick_test else 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'beta_schedule': 'linear',
        'checkpoint_dir': f'{args.checkpoint_dir}/diffusion',
        'log_dir': f'{args.log_dir}/diffusion',
        'save_freq': max(1, epochs // 5),
        'use_ema': True,
        'ema_decay': 0.995,
        'loss_weights': {
            'diffusion': 1.0,
            'track': 0.5,
            'intensity': 0.3,
            'physics': 0.2,
            'consistency': 0.1
        }
    }
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    print(f"\nTraining diffusion model for {epochs} epochs...")
    print("  This may take a while...")
    trainer.train(epochs=epochs)
    
    print_progress(f"Training complete! Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {config['checkpoint_dir']}/best.pth")
    
    return model


def evaluate_and_visualize(args, autoencoder, diffusion):
    """Evaluate model and create visualizations"""
    print_header("STEP 4: EVALUATION AND VISUALIZATION")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = TyphoonDataset(args.output_dir, split='test')
    print_progress(f"Test: {len(test_dataset)} samples")
    
    # Load best models
    print("\nLoading best checkpoints...")
    autoencoder.load_state_dict(
        torch.load(f'{args.checkpoint_dir}/autoencoder/best.pth', 
                   map_location=device)['model_state_dict']
    )
    diffusion.load_state_dict(
        torch.load(f'{args.checkpoint_dir}/diffusion/best.pth',
                   map_location=device)['model_state_dict']
    )
    autoencoder = autoencoder.to(device)
    diffusion = diffusion.to(device)
    autoencoder.eval()
    diffusion.eval()
    print_progress("Models loaded")
    
    # Create predictor
    num_steps = 20 if args.quick_test else args.num_inference_steps
    predictor = TyphoonPredictor(
        autoencoder=autoencoder,
        diffusion_model=diffusion,
        device=device,
        num_inference_steps=num_steps
    )
    print_progress(f"Predictor created (inference steps: {num_steps})")
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    metrics_calculator = PredictionMetrics()
    
    all_metrics = []
    all_predictions = []
    
    n_eval = min(len(test_dataset), args.eval_samples)
    
    for i, batch in enumerate(test_loader):
        if i >= n_eval:
            break
            
        # Make prediction
        with torch.no_grad():
            predictions = predictor.predict(
                past_frames=batch['past_frames'].to(device),
                past_track=batch['track_past'].to(device),
                past_intensity=batch['intensity_past'].to(device)
            )
        
        # Compute metrics
        metrics = metrics_calculator.compute_all_metrics(
            pred_frames=predictions['future_frames'].cpu(),
            true_frames=batch['future_frames'],
            pred_track=predictions['track'].cpu(),
            true_track=batch['track_future'],
            pred_intensity=predictions['intensity'].cpu(),
            true_intensity=batch['intensity_future']
        )
        
        all_metrics.append(metrics)
        all_predictions.append({
            'pred_frames': predictions['future_frames'].cpu().numpy(),
            'true_frames': batch['future_frames'].numpy(),
            'pred_track': predictions['track'].cpu().numpy(),
            'true_track': batch['track_future'].numpy(),
            'pred_intensity': predictions['intensity'].cpu().numpy(),
            'true_intensity': batch['intensity_future'].numpy(),
            'past_track': batch['track_past'].numpy(),
            'past_intensity': batch['intensity_past'].numpy(),
            'case_id': batch.get('case_id', [f'test_{i}'])[0]
        })
        
        print(f"  Sample {i+1}/{n_eval}: "
              f"Track Error={metrics['track_error_mean']:.2f}km, "
              f"Intensity MAE={metrics['intensity_mae']:.2f}m/s")
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print_progress("Evaluation complete!")
    print(f"\n  Average Metrics:")
    print(f"    Track Error: {avg_metrics['track_error_mean']:.2f} km")
    print(f"    Intensity MAE: {avg_metrics['intensity_mae']:.2f} m/s")
    print(f"    SSIM: {avg_metrics['ssim']:.3f}")
    print(f"    Physics Validity: {avg_metrics['physics_valid_ratio']:.1%}")
    
    # Save results
    results_summary = {
        'test_date': datetime.now().isoformat(),
        'data_source': 'IBTrACS_WP',
        'meteorological_data': 'ERA5' if args.use_era5 else 'Synthetic',
        'year_range': f'{args.start_year}-{args.end_year}',
        'n_train_samples': args.n_samples,
        'n_test_samples': n_eval,
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'model_config': {
            'latent_channels': args.latent_channels,
            'hidden_dim': args.hidden_dim,
            'autoencoder_epochs': args.autoencoder_epochs if not args.quick_test else 3,
            'diffusion_epochs': args.diffusion_epochs if not args.quick_test else 3,
        }
    }
    
    results_file = Path(args.results_dir) / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print_progress(f"Results saved to {results_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    viz = TyphoonVisualizer(output_dir=args.results_dir)
    
    n_viz = min(5, len(all_predictions))
    for i in range(n_viz):
        pred = all_predictions[i]
        
        # Trajectory
        viz.plot_trajectory_comparison(
            past_track=pred['past_track'][0],
            true_track=pred['true_track'][0],
            pred_track=pred['pred_track'][0],
            past_intensity=pred['past_intensity'][0],
            true_intensity=pred['true_intensity'][0],
            pred_intensity=pred['pred_intensity'][0],
            save_name=f'trajectory_sample_{i+1}.png'
        )
        
        # Intensity
        viz.plot_intensity_comparison(
            past_intensity=pred['past_intensity'][0],
            true_intensity=pred['true_intensity'][0],
            pred_intensity=pred['pred_intensity'][0],
            save_name=f'intensity_sample_{i+1}.png'
        )
        
        print_progress(f"Visualization {i+1}/{n_viz} complete")
    
    # Error statistics
    viz.plot_error_statistics(all_metrics, save_name='error_statistics.png')
    print_progress("Error statistics plot created")
    
    # Create HTML report
    viz.create_comprehensive_report(
        predictions=all_predictions[:n_viz],
        metrics=all_metrics[:n_viz],
        avg_metrics=avg_metrics
    )
    print_progress(f"HTML report created: {args.results_dir}/prediction_report.html")
    
    return avg_metrics


def main():
    """Main pipeline execution"""
    args = parse_args()
    
    print_header("TYPHOON PREDICTION PIPELINE - REAL DATA")
    print(f"Start Time: {datetime.now()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE - Minimal training for testing")
        args.n_samples = min(args.n_samples, 20)
        args.autoencoder_epochs = 3
        args.diffusion_epochs = 3
        args.eval_samples = 3
    
    # Setup
    setup_directories(args)
    
    # Step 1: Generate real data
    if not args.skip_data_generation and not args.eval_only:
        n_samples = generate_real_data(args)
        if n_samples == 0:
            print("✗ No samples generated. Exiting.")
            sys.exit(1)
    
    # Step 2: Train autoencoder
    if not args.skip_autoencoder and not args.eval_only:
        autoencoder = train_autoencoder(args)
    else:
        # Load existing autoencoder
        print_header("LOADING EXISTING AUTOENCODER")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = f'{args.checkpoint_dir}/autoencoder/best.pth'
        if not os.path.exists(checkpoint_path):
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        autoencoder = SpatialAutoencoder(
            in_channels=config.get('in_channels', 48),
            latent_channels=config.get('latent_channels', args.latent_channels),
            hidden_dims=config.get('hidden_dims', [64, 128, 256, 256]),
            use_attention=config.get('use_attention', True)
        ).to(device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print_progress(f"Loaded from {checkpoint_path}")
    
    # Step 3: Train diffusion
    if not args.skip_diffusion and not args.eval_only:
        diffusion = train_diffusion(args, autoencoder)
    else:
        # Load existing diffusion
        print_header("LOADING EXISTING DIFFUSION MODEL")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = f'{args.checkpoint_dir}/diffusion/best.pth'
        if not os.path.exists(checkpoint_path):
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        diffusion = PhysicsInformedDiffusionModel(
            latent_channels=args.latent_channels,
            hidden_dim=args.hidden_dim,
            num_heads=4,
            depth=3,
            output_frames=8
        ).to(device)
        diffusion.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        print_progress(f"Loaded from {checkpoint_path}")
    
    # Step 4: Evaluate and visualize
    avg_metrics = evaluate_and_visualize(args, autoencoder, diffusion)
    
    # Final summary
    print_header("PIPELINE COMPLETE!")
    print(f"End Time: {datetime.now()}")
    print(f"\n✓ All steps completed successfully!")
    print(f"\nFinal Results:")
    print(f"  Data Source: IBTrACS Western Pacific ({args.start_year}-{args.end_year})")
    print(f"  Meteorological Data: {'ERA5' if args.use_era5 else 'Synthetic'}")
    print(f"  Training Samples: {args.n_samples}")
    print(f"  Track Error: {avg_metrics['track_error_mean']:.2f} km")
    print(f"  Intensity MAE: {avg_metrics['intensity_mae']:.2f} m/s")
    print(f"  SSIM: {avg_metrics['ssim']:.3f}")
    print(f"\nOutputs:")
    print(f"  Data: {args.output_dir}/")
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print(f"  Results: {args.results_dir}/")
    print(f"  Report: {args.results_dir}/prediction_report.html")
    print(f"\nView results:")
    print(f"  open {args.results_dir}/prediction_report.html")
    print("=" * 80)


if __name__ == '__main__':
    main()

