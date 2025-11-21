from imagen_pytorch import Imagen
import inspect

sig = inspect.signature(Imagen.__init__)
params = [p for p in sig.parameters.keys() if 'cond' in p.lower() or 'continuous' in p.lower() or 'embed' in p.lower()]
print("Imagen parameters related to conditioning:")
for p in params:
    print(f"  - {p}")

# Try creating Imagen with condition_on_continuous
try:
    from imagen_pytorch import Unet3D
    unet = Unet3D(dim=32, channels=24, cond_dim=1024)
    imagen = Imagen(
        unets=[unet],
        image_sizes=64,
        timesteps=250,
        condition_on_continuous=True,
        continuous_embed_dim=1024,
    )
    print("\n[OK] Imagen created successfully with condition_on_continuous")
except Exception as e:
    print(f"\n[ERROR] {e}")
    # Try without condition_on_continuous
    try:
        imagen = Imagen(
            unets=[unet],
            image_sizes=64,
            timesteps=250,
        )
        print("[OK] Imagen created without condition_on_continuous")
    except Exception as e2:
        print(f"[ERROR] {e2}")

