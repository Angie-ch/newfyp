"""Check if dataset uses real ERA5 data"""
import numpy as np
from pathlib import Path

# Check a sample file
train_dir = Path('data/processed_temporal_split/train/cases')
files = [f for f in train_dir.glob('*.npz') if not f.name.startswith('._')]

if files:
    f = np.load(files[0], allow_pickle=True)
    print(f"Sample file: {files[0].name}")
    print(f"past_frames shape: {f['past_frames'].shape}")
    print(f"past_frames range: {f['past_frames'].min():.6f} to {f['past_frames'].max():.6f}")
    print(f"Has non-zero data: {(f['past_frames'] != 0).any()}")
    print(f"Non-zero percentage: {(f['past_frames'] != 0).sum() / f['past_frames'].size * 100:.2f}%")
    print(f"\nSample values (first frame, first channel, center 5x5):")
    print(f['past_frames'][0, 0, 30:35, 30:35])
else:
    print("No files found")

