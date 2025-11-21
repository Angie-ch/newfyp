import sys
sys.path.append('forecast-video-diffmodels/imagen')
try:
    from imagen_pytorch import Unet3D, Imagen
    print("[OK] Successfully imported Unet3D and Imagen")
    print(f"Unet3D signature: {Unet3D.__init__.__code__.co_varnames}")
except Exception as e:
    print(f"[ERROR] {e}")

