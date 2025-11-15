import torch
from pathlib import Path

checkpoints = {
    'joint_autoencoder': 'checkpoints/joint_autoencoder/best.pth',
    'joint_diffusion': 'checkpoints/joint_diffusion/best.pth',
    'autoencoder': 'checkpoints/autoencoder/best.pth',
    'diffusion': 'checkpoints/diffusion/best.pth',
}

print("=" * 60)
print("Best Checkpoint Files Information")
print("=" * 60)

for name, path in checkpoints.items():
    if Path(path).exists():
        try:
            ckpt = torch.load(path, map_location='cpu')
            epoch = ckpt.get('epoch', 'N/A')
            val_loss = ckpt.get('val_loss', 'N/A')
            file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
            print(f"\n{name.upper()}:")
            print(f"  Path: {path}")
            print(f"  Size: {file_size:.1f} MB")
            print(f"  Epoch: {epoch}")
            print(f"  Val Loss: {val_loss}")
        except Exception as e:
            print(f"\n{name.upper()}:")
            print(f"  Path: {path}")
            print(f"  Error loading: {e}")
    else:
        print(f"\n{name.upper()}:")
        print(f"  Path: {path}")
        print(f"  File not found")

print("\n" + "=" * 60)
print("RECOMMENDED FOR INFERENCE:")
print("=" * 60)
print("  Autoencoder: checkpoints/joint_autoencoder/best.pth")
print("  Diffusion:   checkpoints/joint_diffusion/best.pth")
print("=" * 60)


