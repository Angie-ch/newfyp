from imagen_pytorch import Unet3D
import inspect

# Create a test Unet3D
unet = Unet3D(dim=32, channels=24, cond_dim=1024)

# Check forward signature
sig = inspect.signature(unet.forward)
print("Unet3D.forward signature:")
print(sig)

# Try calling with different parameters
import torch
x = torch.randn(1, 24, 12, 64, 64)  # (B, C, T, H, W)
time = torch.randint(0, 250, (1,)).long()
cond = torch.randn(1, 1024)

print("\nTrying different parameter names:")
try:
    out = unet(x, time=time, cond=cond)
    print("[OK] unet(x, time=time, cond=cond) works")
except Exception as e:
    print(f"[ERROR] cond: {e}")

try:
    out = unet(x, time=time, text_embeds=cond)
    print("[OK] unet(x, time=time, text_embeds=cond) works")
except Exception as e:
    print(f"[ERROR] text_embeds: {e}")

try:
    out = unet(x, time=time)
    print("[OK] unet(x, time=time) works (no condition)")
except Exception as e:
    print(f"[ERROR] no cond: {e}")

