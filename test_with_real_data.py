"""
Test typhoon prediction pipeline with real data and visualizations
"""

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.real_data_loader import IBTrACSLoader
from data.datasets.typhoon_dataset import TyphoonDataset
from models.autoencoder.autoencoder import SpatialAutoencoder
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel
from training.trainers.autoencoder_trainer import AutoencoderTrainer
from training.trainers.diffusion_trainer import DiffusionTrainer
from visualize_results import TyphoonVisualizer


def main():
    print("="*80)
    print("TYPHOON PREDICTION WITH REAL DATA AND VISUALIZATIONS")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = {
        'image_size': (64, 64),
        'n_channels': 40,
        'past_timesteps': 12,
        'future_timesteps': 8,
        'n_samples': 50,  # Smaller for faster testing
        'batch_size': 4,
        'autoencoder_epochs': 5,
        'diffusion_epochs': 5,
        'start_year': 2018,
        'end_year': 2023,
    }
    
    # Create directories
    Path("checkpoints/autoencoder").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/diffusion").mkdir(parents=True, exist_ok=True)
    Path("visualizations").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: LOAD REAL DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING REAL TYPHOON DATA FROM IBTRACS")
    print("="*80)
    
    try:
        loader = IBTrACSLoader(data_dir="data/raw")
        
        # Check if data already exists
        cache_file = Path("data/processed/real_typhoons.npz")
        
        if cache_file.exists():
            print(f"Loading cached dataset from {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            samples = data['samples'].tolist()
            print(f"✓ Loaded {len(samples)} cached samples")
        else:
            print("Creating new dataset from IBTrACS...")
            samples = loader.create_dataset(
                n_samples=config['n_samples'],
                start_year=config['start_year'],
                end_year=config['end_year'],
                past_timesteps=config['past_timesteps'],
                future_timesteps=config['future_timesteps'],
                save_path=cache_file
            )
        
        if len(samples) == 0:
            raise ValueError("No samples created")
        
        print(f"\n✓ Dataset ready: {len(samples)} samples")
        print(f"  Sample storms: {samples[0]['storm_name']}, {samples[1]['storm_name']}, ...")
        
    except Exception as e:
        print(f"\n✗ Error loading real data: {e}")
        print("\nFalling back to synthetic data for testing...")
        
        from test_pipeline import generate_synthetic_data
        train_data, val_data, test_data = generate_synthetic_data(
            n_cases=config['n_samples'],
            past_timesteps=config['past_timesteps'],
            future_timesteps=config['future_timesteps']
        )
        
        # Combine for compatibility
        samples = train_data + val_data + test_data
        print(f"✓ Generated {len(samples)} synthetic samples")
    
    # ========================================================================
    # STEP 2: CREATE DATASETS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: CREATING PYTORCH DATASETS")
    print("="*80)
    
    # Split data
    n_train = int(0.7 * len(samples))
    n_val = int(0.15 * len(samples))
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    train_dataset = TyphoonDataset(train_samples, split='train')
    val_dataset = TyphoonDataset(val_samples, split='val')
    test_dataset = TyphoonDataset(test_samples, split='test')
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    
    # ========================================================================
    # STEP 3: TRAIN AUTOENCODER
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING AUTOENCODER")
    print("="*80)
    
    autoencoder = SpatialAutoencoder(
        in_channels=config['n_channels'],
        latent_channels=8
    ).to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in autoencoder.parameters()):,} parameters")
    
    ae_trainer = AutoencoderTrainer(
        model=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir="checkpoints/autoencoder"
    )
    
    print(f"\nTraining for {config['autoencoder_epochs']} epochs...")
    ae_trainer.train(epochs=config['autoencoder_epochs'])
    
    print(f"\n✓ Autoencoder training completed")
    print(f"  Best validation loss: {ae_trainer.best_val_loss:.4f}")
    
    # Load best model
    best_ae_path = Path("checkpoints/autoencoder/best.pth")
    if best_ae_path.exists():
        checkpoint = torch.load(best_ae_path, map_location=device, weights_only=False)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best autoencoder checkpoint")
    
    # ========================================================================
    # STEP 4: TRAIN DIFFUSION MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING DIFFUSION MODEL")
    print("="*80)
    
    diffusion_model = PhysicsInformedDiffusionModel(
        latent_channels=8,
        output_frames=config['future_timesteps']
    ).to(device)
    
    print(f"✓ Model created with {sum(p.numel() for p in diffusion_model.parameters()):,} parameters")
    
    diff_trainer = DiffusionTrainer(
        model=diffusion_model,
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir="checkpoints/diffusion"
    )
    
    print(f"\nTraining for {config['diffusion_epochs']} epochs...")
    diff_trainer.train(epochs=config['diffusion_epochs'])
    
    print(f"\n✓ Diffusion model training completed")
    print(f"  Best validation loss: {diff_trainer.best_val_loss:.4f}")
    
    # Load best model
    best_diff_path = Path("checkpoints/diffusion/best.pth")
    if best_diff_path.exists():
        checkpoint = torch.load(best_diff_path, map_location=device, weights_only=False)
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best diffusion model checkpoint")
    
    # ========================================================================
    # STEP 5: GENERATE PREDICTIONS AND VISUALIZE
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: GENERATING PREDICTIONS AND VISUALIZATIONS")
    print("="*80)
    
    visualizer = TyphoonVisualizer(output_dir="visualizations")
    
    # Set models to eval mode
    autoencoder.eval()
    diffusion_model.eval()
    
    # Collect predictions
    all_track_errors = []
    all_intensity_errors = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:  # Visualize first 5 samples
                break
            
            past_frames = batch['past_frames'].to(device)
            future_frames = batch['future_frames'].to(device)
            track_future = batch['track_future'].to(device)
            intensity_future = batch['intensity_future'].to(device)
            
            # Generate predictions
            # Encode past frames
            past_encoded = autoencoder.encode(past_frames)  # (B, T, latent_dim, H', W')
            
            # Predict future in latent space
            predictions = diffusion_model(past_encoded)
            pred_latent = predictions['frames']  # (B, T_future, latent_dim, H', W')
            pred_track = predictions['track']  # (B, T_future, 2)
            pred_intensity = predictions['intensity']  # (B, T_future)
            
            # Decode predictions
            B, T_future, C, H, W = pred_latent.shape
            pred_latent_flat = pred_latent.reshape(B * T_future, C, H, W)
            pred_frames_flat = autoencoder.decode(pred_latent_flat)
            pred_frames = pred_frames_flat.reshape(B, T_future, *pred_frames_flat.shape[1:])
            
            # Convert to numpy for visualization
            past_frames_np = past_frames[0].cpu().numpy()
            future_frames_np = future_frames[0].cpu().numpy()
            pred_frames_np = pred_frames[0].cpu().numpy()
            
            past_track_np = batch['track_past'][0].cpu().numpy()
            future_track_np = track_future[0].cpu().numpy()
            pred_track_np = pred_track[0].cpu().numpy()
            
            past_intensity_np = batch['intensity_past'][0].cpu().numpy()
            future_intensity_np = intensity_future[0].cpu().numpy()
            pred_intensity_np = pred_intensity[0].cpu().numpy()
            
            # Create visualizations for this sample
            print(f"\n  Sample {i+1}:")
            
            # Trajectory
            track_error = visualizer.plot_trajectory_comparison(
                past_track_np, future_track_np, pred_track_np,
                save_name=f"trajectory_sample_{i+1}.png"
            )
            all_track_errors.append(track_error)
            print(f"    Track error: {track_error:.1f} km")
            
            # Intensity
            intensity_metrics = visualizer.plot_intensity_comparison(
                past_intensity_np, future_intensity_np, pred_intensity_np,
                save_name=f"intensity_sample_{i+1}.png"
            )
            all_intensity_errors.append(intensity_metrics['rmse'])
            print(f"    Intensity RMSE: {intensity_metrics['rmse']:.2f} m/s")
            
            # Frames
            visualizer.plot_frames_comparison(
                past_frames_np, future_frames_np, pred_frames_np,
                save_name=f"frames_sample_{i+1}.png"
            )
            
            # Animation
            combined_frames = np.concatenate([past_frames_np, pred_frames_np], axis=0)
            combined_track = np.concatenate([past_track_np, pred_track_np], axis=0)
            combined_intensity = np.concatenate([past_intensity_np, pred_intensity_np], axis=0)
            
            visualizer.create_animation(
                combined_frames, combined_track, combined_intensity,
                save_name=f"animation_sample_{i+1}.gif"
            )
    
    # ========================================================================
    # STEP 6: SUMMARY STATISTICS AND REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: CREATING SUMMARY REPORT")
    print("="*80)
    
    # Plot training curves
    if hasattr(ae_trainer, 'train_losses') and len(ae_trainer.train_losses) > 0:
        visualizer.plot_training_curves(
            ae_trainer.train_losses,
            ae_trainer.val_losses,
            model_name="Autoencoder",
            save_name="autoencoder_training_curves.png"
        )
    
    if hasattr(diff_trainer, 'train_losses') and len(diff_trainer.train_losses) > 0:
        visualizer.plot_training_curves(
            diff_trainer.train_losses,
            diff_trainer.val_losses,
            model_name="Diffusion Model",
            save_name="diffusion_training_curves.png"
        )
    
    # Error statistics
    if len(all_track_errors) > 0:
        visualizer.plot_error_statistics(
            all_track_errors,
            all_intensity_errors
        )
        
        # Print summary
        print("\n📊 PREDICTION PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Mean Track Error: {np.mean(all_track_errors):.1f} ± {np.std(all_track_errors):.1f} km")
        print(f"Mean Intensity RMSE: {np.mean(all_intensity_errors):.2f} ± {np.std(all_intensity_errors):.2f} m/s")
        print(f"Samples Evaluated: {len(all_track_errors)}")
    
    # Create HTML report
    results = {
        'timestamp': datetime.now().isoformat(),
        'mean_track_error': float(np.mean(all_track_errors)) if all_track_errors else 0,
        'mean_intensity_rmse': float(np.mean(all_intensity_errors)) if all_intensity_errors else 0,
        'n_samples': len(all_track_errors),
        'config': config
    }
    
    visualizer.create_comprehensive_report(results)
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("✓ PIPELINE TEST COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now()}")
    print("\nResults saved to:")
    print(f"  - Visualizations: visualizations/")
    print(f"  - Checkpoints: checkpoints/")
    print(f"  - Report: visualizations/prediction_report.html")
    print("\nOpen the HTML report in your browser to see all results!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

