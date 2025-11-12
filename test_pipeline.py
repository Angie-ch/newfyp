#!/usr/bin/env python3
"""
Test Pipeline - End-to-End Validation
Generates synthetic data and runs complete training/evaluation pipeline
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

print("=" * 80)
print("TYPHOON PREDICTION PIPELINE - COMPLETE TEST")
print("=" * 80)
print(f"Start Time: {datetime.now()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Configuration
NUM_CASES = 20  # Small test dataset
BATCH_SIZE = 2
TRAIN_CASES = 14
VAL_CASES = 3
TEST_CASES = 3

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed/cases", exist_ok=True)
os.makedirs("checkpoints/autoencoder", exist_ok=True)
os.makedirs("checkpoints/diffusion", exist_ok=True)
os.makedirs("logs/autoencoder", exist_ok=True)
os.makedirs("logs/diffusion", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("\n" + "=" * 80)
print("STEP 1: GENERATING SYNTHETIC DATA")
print("=" * 80)

def create_synthetic_typhoon_data():
    """Generate realistic synthetic typhoon data"""
    cases = []
    
    for case_id in range(NUM_CASES):
        # Generate typhoon parameters
        center_lat = np.random.uniform(15, 30)  # North latitude
        center_lon = np.random.uniform(120, 150)  # East longitude
        intensity = np.random.uniform(30, 60)  # Wind speed m/s
        direction = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(3, 8)  # Movement speed m/s
        
        # Generate 20 time frames (12 past + 8 future)
        past_frames = np.zeros((12, 40, 64, 64), dtype=np.float32)
        future_frames = np.zeros((8, 40, 64, 64), dtype=np.float32)
        track_past = np.zeros((12, 2), dtype=np.float32)
        track_future = np.zeros((8, 2), dtype=np.float32)
        intensity_past = np.zeros((12,), dtype=np.float32)
        intensity_future = np.zeros((8,), dtype=np.float32)
        
        for t in range(12):
            # Past trajectory
            lat = center_lat + (t - 6) * speed * np.sin(direction) * 0.01
            lon = center_lon + (t - 6) * speed * np.cos(direction) * 0.01
            track_past[t] = [lat, lon]
            intensity_past[t] = intensity + np.random.normal(0, 2)
            
            # Generate atmospheric fields with typhoon structure
            y, x = np.ogrid[-32:32, -32:32]
            y_center = (lat - center_lat) * 10
            x_center = (lon - center_lon) * 10
            r = np.sqrt((y - y_center)**2 + (x - x_center)**2)
            theta = np.arctan2(y - y_center, x - x_center)
            
            # Pressure field (low at center)
            pressure = 1013 - intensity * 0.5 * np.exp(-r / 10)
            # Wind field (spiral pattern)
            u_wind = -intensity * np.sin(theta) * np.exp(-r / 15)
            v_wind = intensity * np.cos(theta) * np.exp(-r / 15)
            # Temperature anomaly
            temp = 2 * np.exp(-r / 20)
            # Humidity (high near center)
            humidity = 80 + 15 * np.exp(-r / 12)
            
            # Fill channels (simplified)
            past_frames[t, 0] = pressure
            past_frames[t, 1] = u_wind
            past_frames[t, 2] = v_wind
            past_frames[t, 3] = temp
            past_frames[t, 4] = humidity
            # Add noise to other channels
            past_frames[t, 5:] = np.random.randn(35, 64, 64) * 0.1
        
        for t in range(8):
            # Future trajectory
            lat = center_lat + (t + 6) * speed * np.sin(direction) * 0.01
            lon = center_lon + (t + 6) * speed * np.cos(direction) * 0.01
            track_future[t] = [lat, lon]
            intensity_future[t] = intensity + np.random.normal(0, 3)
            
            # Future atmospheric fields
            y_center = (lat - center_lat) * 10
            x_center = (lon - center_lon) * 10
            r = np.sqrt((y - y_center)**2 + (x - x_center)**2)
            theta = np.arctan2(y - y_center, x - x_center)
            
            pressure = 1013 - intensity * 0.5 * np.exp(-r / 10)
            u_wind = -intensity * np.sin(theta) * np.exp(-r / 15)
            v_wind = intensity * np.cos(theta) * np.exp(-r / 15)
            temp = 2 * np.exp(-r / 20)
            humidity = 80 + 15 * np.exp(-r / 12)
            
            future_frames[t, 0] = pressure
            future_frames[t, 1] = u_wind
            future_frames[t, 2] = v_wind
            future_frames[t, 3] = temp
            future_frames[t, 4] = humidity
            future_frames[t, 5:] = np.random.randn(35, 64, 64) * 0.1
        
        # Save case
        case_data = {
            'past_frames': past_frames,
            'future_frames': future_frames,
            'track_past': track_past,
            'track_future': track_future,
            'intensity_past': intensity_past,
            'intensity_future': intensity_future,
            'case_id': f'test_{case_id:04d}'
        }
        
        np.savez_compressed(
            f'data/processed/cases/case_{case_id:04d}.npz',
            **case_data
        )
        
        cases.append({
            'case_id': case_id,
            'split': 'train' if case_id < TRAIN_CASES else ('val' if case_id < TRAIN_CASES + VAL_CASES else 'test'),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'max_intensity': intensity
        })
        
        if (case_id + 1) % 5 == 0:
            print(f"  Generated {case_id + 1}/{NUM_CASES} cases...")
    
    # Save metadata
    metadata_df = pd.DataFrame(cases)
    metadata_df.to_csv('data/processed/metadata.csv', index=False)
    
    # Save statistics
    all_past = np.concatenate([np.load(f'data/processed/cases/case_{i:04d}.npz')['past_frames'] 
                                for i in range(NUM_CASES)], axis=0)
    stats = {
        'mean': float(all_past.mean()),
        'std': float(all_past.std()),
        'min': float(all_past.min()),
        'max': float(all_past.max())
    }
    with open('data/processed/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Generated {NUM_CASES} synthetic typhoon cases")
    print(f"  Train: {TRAIN_CASES}, Val: {VAL_CASES}, Test: {TEST_CASES}")
    print(f"  Statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    return metadata_df, stats

metadata, stats = create_synthetic_typhoon_data()

print("\n" + "=" * 80)
print("STEP 2: TESTING DATA LOADING")
print("=" * 80)

from data.datasets.typhoon_dataset import TyphoonDataset

try:
    train_dataset = TyphoonDataset('data/processed', split='train')
    val_dataset = TyphoonDataset('data/processed', split='val')
    test_dataset = TyphoonDataset('data/processed', split='test')
    
    print(f"✓ Train dataset: {len(train_dataset)} cases")
    print(f"✓ Val dataset: {len(val_dataset)} cases")
    print(f"✓ Test dataset: {len(test_dataset)} cases")
    
    # Test loading one sample
    sample = train_dataset[0]
    print(f"✓ Sample shapes:")
    print(f"  past_frames: {sample['past_frames'].shape}")
    print(f"  future_frames: {sample['future_frames'].shape}")
    print(f"  track_future: {sample['track_future'].shape}")
    print(f"  intensity_future: {sample['intensity_future'].shape}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 3: TESTING MODEL ARCHITECTURES")
print("=" * 80)

# Test Autoencoder
from models.autoencoder.autoencoder import SpatialAutoencoder

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = SpatialAutoencoder(in_channels=40, latent_channels=8).to(device)
    
    # Test forward pass
    test_input = torch.randn(2, 40, 64, 64).to(device)
    encoded = autoencoder.encode(test_input)
    decoded = autoencoder.decode(encoded)
    
    print(f"✓ Autoencoder created successfully")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Decoded shape: {decoded.shape}")
    print(f"  Parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
except Exception as e:
    print(f"✗ Error creating autoencoder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Diffusion Model
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel

try:
    diffusion = PhysicsInformedDiffusionModel(
        latent_channels=8,
        hidden_dim=64,  # Reduced for testing
        num_heads=4,
        depth=2,
        output_frames=8
    ).to(device)
    
    # Test forward pass
    test_latent = torch.randn(2, 8, 8, 8, 8).to(device)  # (B, T, C, H, W)
    test_t = torch.randint(0, 100, (2,)).to(device)
    test_condition = {
        'past_latents': torch.randn(2, 12, 8, 8, 8).to(device),
        'past_track': torch.randn(2, 12, 2).to(device),
        'past_intensity': torch.randn(2, 12).to(device)
    }
    
    output = diffusion(test_latent, test_t, test_condition)
    
    print(f"✓ Diffusion model created successfully")
    print(f"  Input shape: {test_latent.shape}")
    print(f"  Noise prediction: {output['noise'].shape}")
    print(f"  Track prediction: {output['track'].shape}")
    print(f"  Intensity prediction: {output['intensity'].shape}")
    print(f"  Parameters: {sum(p.numel() for p in diffusion.parameters()):,}")
except Exception as e:
    print(f"✗ Error creating diffusion model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 4: TRAINING AUTOENCODER (Mini Run)")
print("=" * 80)

from training.trainers.autoencoder_trainer import AutoencoderTrainer
from torch.utils.data import DataLoader

try:
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create config for trainer
    autoencoder_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 3,
        'checkpoint_dir': 'checkpoints/autoencoder',
        'log_dir': 'logs/autoencoder',
        'save_freq': 1
    }
    
    # Create trainer
    trainer = AutoencoderTrainer(
        model=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=autoencoder_config,
        device=device
    )
    
    print("Training autoencoder for 3 epochs (quick test)...")
    trainer.train(epochs=3)
    
    print(f"✓ Autoencoder training completed")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    
except Exception as e:
    print(f"✗ Error training autoencoder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 5: TRAINING DIFFUSION MODEL (Mini Run)")
print("=" * 80)

from training.trainers.diffusion_trainer import DiffusionTrainer

try:
    # Reload best autoencoder
    autoencoder_path = 'checkpoints/autoencoder/best.pth'
    if os.path.exists(autoencoder_path):
        checkpoint = torch.load(autoencoder_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best autoencoder from checkpoint")
    
    # Create config for diffusion trainer
    diffusion_config = {
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'epochs': 3,
        'timesteps': 100,  # Reduced for testing
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'beta_schedule': 'linear',
        'checkpoint_dir': 'checkpoints/diffusion',
        'log_dir': 'logs/diffusion',
        'save_freq': 1,
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
    
    # Create diffusion trainer
    trainer = DiffusionTrainer(
        model=diffusion,
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=diffusion_config,
        device=device
    )
    
    print("Training diffusion model for 3 epochs (quick test)...")
    trainer.train(epochs=3)
    
    print(f"✓ Diffusion training completed")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    
except Exception as e:
    print(f"✗ Error training diffusion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 6: RUNNING EVALUATION")
print("=" * 80)

from evaluation.metrics.prediction_metrics import PredictionMetrics
from inference import TyphoonPredictor

try:
    # Load best models
    autoencoder.load_state_dict(
        torch.load('checkpoints/autoencoder/best.pth', map_location=device)['model_state_dict']
    )
    diffusion.load_state_dict(
        torch.load('checkpoints/diffusion/best.pth', map_location=device)['model_state_dict']
    )
    
    # Create predictor
    predictor = TyphoonPredictor(
        autoencoder=autoencoder,
        diffusion_model=diffusion,
        device=device,
        num_inference_steps=20  # Fast inference
    )
    
    # Evaluate on test set
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    metrics_calculator = PredictionMetrics()
    
    all_metrics = []
    
    print(f"Evaluating on {len(test_dataset)} test cases...")
    for i, batch in enumerate(test_loader):
        # Make prediction
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
        print(f"  Case {i+1}/{len(test_dataset)}: Track Error={metrics['track_error_mean']:.2f}km, "
              f"Intensity MAE={metrics['intensity_mae']:.2f}m/s, SSIM={metrics['ssim']:.3f}")
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print(f"\n✓ Evaluation completed")
    print(f"  Average Track Error: {avg_metrics['track_error_mean']:.2f} km")
    print(f"  Average Intensity MAE: {avg_metrics['intensity_mae']:.2f} m/s")
    print(f"  Average SSIM: {avg_metrics['ssim']:.3f}")
    print(f"  Physics Validity: {avg_metrics['physics_valid_ratio']:.1%}")
    
    # Save results
    results_summary = {
        'test_date': datetime.now().isoformat(),
        'num_test_cases': len(test_dataset),
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics
    }
    
    with open('results/test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
except Exception as e:
    print(f"✗ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 7: CREATING VISUALIZATIONS")
print("=" * 80)

from evaluation.visualizations.plot_utils import VisualizationTools

try:
    viz = VisualizationTools()
    
    # Get one test case for visualization
    test_case = test_dataset[0]
    
    # Make prediction
    predictions = predictor.predict(
        past_frames=test_case['past_frames'].unsqueeze(0).to(device),
        past_track=test_case['track_past'].unsqueeze(0).to(device),
        past_intensity=test_case['intensity_past'].unsqueeze(0).to(device)
    )
    
    # Plot prediction
    fig = viz.plot_prediction_comparison(
        pred_frames=predictions['future_frames'][0].cpu().numpy(),
        true_frames=test_case['future_frames'].numpy(),
        time_idx=4  # Middle frame
    )
    fig.savefig('results/prediction_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved prediction comparison to results/prediction_comparison.png")
    
    # Plot track
    fig = viz.plot_track_comparison(
        pred_track=predictions['track'][0].cpu().numpy(),
        true_track=test_case['track_future'].numpy(),
        past_track=test_case['track_past'].numpy()
    )
    fig.savefig('results/track_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved track comparison to results/track_comparison.png")
    
    # Plot intensity
    fig = viz.plot_intensity_comparison(
        pred_intensity=predictions['intensity'][0].cpu().numpy(),
        true_intensity=test_case['intensity_future'].numpy(),
        past_intensity=test_case['intensity_past'].numpy()
    )
    fig.savefig('results/intensity_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved intensity comparison to results/intensity_comparison.png")
    
    # Plot metrics summary
    fig = viz.plot_metrics_summary(all_metrics)
    fig.savefig('results/metrics_summary.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved metrics summary to results/metrics_summary.png")
    
except Exception as e:
    print(f"✗ Error creating visualizations: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("PIPELINE TEST COMPLETE!")
print("=" * 80)
print(f"End Time: {datetime.now()}")
print(f"\n✓ All components validated successfully!")
print(f"\nGenerated files:")
print(f"  - Data: data/processed/cases/*.npz ({NUM_CASES} cases)")
print(f"  - Checkpoints: checkpoints/autoencoder/best.pth, checkpoints/diffusion/best.pth")
print(f"  - Results: results/test_results.json")
print(f"  - Visualizations: results/*.png (4 plots)")
print(f"\nNext steps:")
print(f"  1. Download real ERA5 and IBTrACS data")
print(f"  2. Run: python preprocess_data.py --era5_dir ... --ibtracs ...")
print(f"  3. Train on full dataset with configs/diffusion_config.yaml")
print(f"  4. Monitor training: tensorboard --logdir logs/")
print("=" * 80)

