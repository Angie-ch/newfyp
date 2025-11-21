"""
Evaluate Typhoon Prediction Model
Visualize predictions and compute metrics
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import json

print("=" * 80)
print("TYPHOON TRAJECTORY PREDICTION - EVALUATION")
print("=" * 80)

# Import model (same as training)
from train_typhoon_prediction import TyphoonPredictor, TyphoonDataset

# Configuration
DATA_DIR = Path("D:/typhoon_data_2018_2021_full")
MODEL_PATH = "best_typhoon_model.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")

# Load model
print("\n[1] Loading model...")
model = TyphoonPredictor(input_channels=24, hidden_channels=64).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.2f})")

# Load test dataset
print("\n[2] Loading test dataset...")
test_dataset = TyphoonDataset(DATA_DIR, 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluation
print("\n[3] Evaluating on test set...")
print("=" * 80)

all_errors = []
all_track_errors = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        past_frames = batch['past_frames'].to(DEVICE)
        future_frames_gt = batch['future_frames'].to(DEVICE)
        track_past = batch['track_past'].to(DEVICE)
        track_future_gt = batch['track_future'].to(DEVICE)
        
        # Predict
        future_frames_pred, track_future_pred = model(past_frames, track_past)
        
        # Compute errors
        frame_error = torch.mean((future_frames_pred - future_frames_gt) ** 2).item()
        track_error = torch.mean((track_future_pred - track_future_gt) ** 2).item()
        
        all_errors.append(frame_error)
        all_track_errors.append(track_error)
        
        # Print per-sample results
        if i < 5:  # Show first 5
            print(f"\nSample {i+1}:")
            print(f"  Frame MSE:  {frame_error:.2f}")
            print(f"  Track MSE:  {track_error:.4f}")
            
            # Compute track error in km (approximate)
            track_error_km = np.sqrt(track_error) * 111  # 1 degree ~ 111 km
            print(f"  Track RMSE: {track_error_km:.2f} km")

# Summary statistics
print("\n" + "=" * 80)
print("TEST SET SUMMARY")
print("=" * 80)
print(f"Total test samples: {len(all_errors)}")
print(f"\nFrame Prediction:")
print(f"  Mean MSE:  {np.mean(all_errors):.2f}")
print(f"  Std MSE:   {np.std(all_errors):.2f}")
print(f"  Min MSE:   {np.min(all_errors):.2f}")
print(f"  Max MSE:   {np.max(all_errors):.2f}")

print(f"\nTrack Prediction:")
print(f"  Mean MSE:  {np.mean(all_track_errors):.4f}")
print(f"  Std MSE:   {np.std(all_track_errors):.4f}")
avg_track_rmse_km = np.sqrt(np.mean(all_track_errors)) * 111
print(f"  Avg RMSE:  {avg_track_rmse_km:.2f} km")

# Visualize a few predictions
print("\n[4] Visualizing predictions...")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Typhoon Trajectory Predictions (Test Set)', fontsize=16)

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= 9:  # Show first 9
            break
        
        past_frames = batch['past_frames'].to(DEVICE)
        track_past = batch['track_past'].to(DEVICE)
        track_future_gt = batch['track_future'].cpu().numpy()[0]
        
        # Predict
        _, track_future_pred = model(past_frames, track_past)
        track_future_pred = track_future_pred.cpu().numpy()[0]
        track_past = track_past.cpu().numpy()[0]
        
        # Plot
        ax = axes[i // 3, i % 3]
        
        # Past track (ground truth)
        ax.plot(track_past[:, 0], track_past[:, 1], 'b.-', label='Past (GT)', linewidth=2, markersize=8)
        
        # Future track (ground truth)
        ax.plot(track_future_gt[:, 0], track_future_gt[:, 1], 'g.-', label='Future (GT)', linewidth=2, markersize=8)
        
        # Future track (predicted)
        ax.plot(track_future_pred[:, 0], track_future_pred[:, 1], 'r.--', label='Future (Pred)', linewidth=2, markersize=8)
        
        # Mark start and end
        ax.plot(track_past[0, 0], track_past[0, 1], 'bs', markersize=12, label='Start')
        ax.plot(track_future_gt[-1, 0], track_future_gt[-1, 1], 'g^', markersize=12, label='End (GT)')
        ax.plot(track_future_pred[-1, 0], track_future_pred[-1, 1], 'r^', markersize=12, label='End (Pred)')
        
        # Compute error
        error_km = np.sqrt(np.mean((track_future_pred - track_future_gt) ** 2)) * 111
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Sample {i+1} (RMSE: {error_km:.1f} km)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('typhoon_predictions.png', dpi=150, bbox_inches='tight')
print(f"  Saved visualization to: typhoon_predictions.png")

# Save metrics
metrics = {
    'num_samples': len(all_errors),
    'frame_mse': {
        'mean': float(np.mean(all_errors)),
        'std': float(np.std(all_errors)),
        'min': float(np.min(all_errors)),
        'max': float(np.max(all_errors)),
    },
    'track_mse': {
        'mean': float(np.mean(all_track_errors)),
        'std': float(np.std(all_track_errors)),
    },
    'track_rmse_km': float(avg_track_rmse_km),
}

with open('evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"  Saved metrics to: evaluation_metrics.json")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print(f"\nKey Result:")
print(f"  Average track prediction error: {avg_track_rmse_km:.2f} km (72-hour forecast)")
print(f"\n  This is the distance error for predictions 72 hours (3 days) into the future!")
print("=" * 80)

