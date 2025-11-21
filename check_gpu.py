"""Check GPU availability"""
import torch

print("=== GPU CHECK ===")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU detected")
    print("Note: GPU is not needed for data regeneration")
    print("GPU will be used for training models")

