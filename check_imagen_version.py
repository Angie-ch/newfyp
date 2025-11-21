"""
检查imagen-pytorch版本和Unet3D支持情况
"""
import sys
import inspect
from imagen_pytorch import Unet3D, Imagen
from imagen_pytorch.imagen_video import Unet3D as Unet3D_video

print("=" * 80)
print("IMAGEN-PYTORCH VERSION CHECK")
print("=" * 80)

# 检查版本
try:
    import imagen_pytorch
    print(f"\n[1] Installed version: {imagen_pytorch.__version__ if hasattr(imagen_pytorch, '__version__') else 'Unknown'}")
    print(f"    Location: {imagen_pytorch.__file__}")
except:
    print("[1] Cannot get version info")

# Check Unet3D
print("\n[2] Unet3D Check:")
print(f"    Unet3D location: {inspect.getfile(Unet3D)}")
print(f"    Unet3D module: {Unet3D.__module__}")

# Check Unet3D forward signature
print("\n[3] Unet3D.forward signature:")
sig = inspect.signature(Unet3D.forward)
print(f"    {sig}")

# Check for video-related parameters
print("\n[4] Unet3D.__init__ parameters (video-related):")
init_sig = inspect.signature(Unet3D.__init__)
video_params = [p for p in init_sig.parameters.keys() if 'video' in p.lower() or 'temporal' in p.lower() or 'time' in p.lower()]
if video_params:
    print(f"    Found video-related params: {video_params}")
else:
    print("    No obvious video-related parameters found")

# Check cond_video_frames support
print("\n[5] Does forward support cond_video_frames:")
forward_params = [p for p in sig.parameters.keys() if 'video' in p.lower() or 'cond' in p.lower()]
print(f"    Related params: {forward_params}")

# Try creating Unet3D
print("\n[6] Test Unet3D creation:")
try:
    unet = Unet3D(
        dim=32,
        channels=24,
        cond_dim=1024,
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
    )
    print("    [OK] Unet3D created successfully")
    
    # Check forward method
    import torch
    x = torch.randn(1, 24, 12, 64, 64)  # (B, C, T, H, W)
    time = torch.randint(0, 250, (1,)).long()
    cond_video = torch.randn(1, 24, 8, 64, 64)  # (B, C, T_cond, H, W)
    
    print("\n[7] Test forward call:")
    try:
        out = unet(x, time=time, cond_video_frames=cond_video)
        print(f"    [OK] forward call successful, output shape: {out.shape}")
    except Exception as e:
        print(f"    [ERROR] forward call failed: {e}")
        print(f"    Error type: {type(e).__name__}")
        
except Exception as e:
    print(f"    [ERROR] Unet3D creation failed: {e}")

print("\n" + "=" * 80)

