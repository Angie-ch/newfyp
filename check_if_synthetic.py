"""Check if current dataset uses synthetic data"""
import numpy as np
from pathlib import Path
import json

# Check dataset_info.json
info_file = Path('data/processed_temporal_split/dataset_info.json')
if info_file.exists():
    info = json.load(open(info_file))
    print("=== DATASET INFO ===")
    print(f"Meteorological data: {info.get('meteorological_data')}")
    print(f"Generation date: {info.get('generation_date')}")
    print(f"Total samples: {info.get('total_samples')}")
    print()

# Check actual sample data
sample_dir = Path('data/processed_temporal_split/train/cases')
if sample_dir.exists():
    samples = [s for s in sample_dir.glob('*.npz') if not s.name.startswith('._')]
    if samples:
        sample = np.load(samples[0], allow_pickle=True)
        frames = sample['past_frames']
        
        non_zero = (frames != 0).sum()
        total = frames.size
        pct = non_zero / total * 100
        
        print("=== SAMPLE DATA ANALYSIS ===")
        print(f"File: {samples[0].name}")
        print(f"Shape: {frames.shape}")
        print(f"Non-zero: {non_zero:,}/{total:,} ({pct:.2f}%)")
        print(f"Range: [{frames.min():.4f}, {frames.max():.4f}]")
        print(f"Mean: {frames.mean():.4f}, Std: {frames.std():.4f}")
        print()
        
        if pct < 1.0:
            print("VERDICT: SYNTHETIC DATA (mostly zeros)")
        elif pct > 50:
            print("VERDICT: REAL ERA5 DATA")
        else:
            print("VERDICT: Mixed - needs manual check")
    else:
        print("No sample files found")
else:
    print("Cases directory doesn't exist")

