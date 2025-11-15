"""Check best model information"""
import torch
from pathlib import Path

best_path = Path('checkpoints/joint_autoencoder/best.pth')
if not best_path.exists():
    print("Best model not found!")
    exit(1)

ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
file_size_mb = best_path.stat().st_size / (1024 * 1024)

print("=" * 70)
print("BEST MODEL INFORMATION")
print("=" * 70)
print(f"\nFile: {best_path}")
print(f"Size: {file_size_mb:.1f} MB")
print(f"\nEpoch: {ckpt.get('epoch', 'N/A')}")
print(f"Val Loss: {ckpt.get('val_loss', 'N/A'):.6f}")

# Check if config exists
if 'config' in ckpt:
    config = ckpt['config']
    print(f"\nTraining Config:")
    print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"  Weight Decay: {config.get('weight_decay', 'N/A')}")

print("\n" + "=" * 70)
print("This is the best model based on validation loss!")
print("=" * 70)

