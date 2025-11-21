"""Verify new dataset files contain real ERA5 data"""
import numpy as np
from pathlib import Path

# Check the newest file
train_dir = Path('data/processed_temporal_split/train/cases')
files = sorted([f for f in train_dir.glob('*.npz') if not f.name.startswith('._')], 
               key=lambda x: x.stat().st_mtime, reverse=True)

if files:
    newest = files[0]
    print(f'Checking newest file: {newest.name}')
    print(f'Modified: {newest.stat().st_mtime}')
    print('')
    
    f = np.load(newest, allow_pickle=True)
    print('VERIFICATION:')
    print(f'  past_frames shape: {f["past_frames"].shape}')
    print(f'  past_frames range: {f["past_frames"].min():.6f} to {f["past_frames"].max():.6f}')
    print(f'  Has non-zero data: {(f["past_frames"] != 0).any()}')
    
    if (f["past_frames"] != 0).any():
        non_zero_pct = (f["past_frames"] != 0).sum() / f["past_frames"].size * 100
        print(f'  Non-zero percentage: {non_zero_pct:.2f}%')
        print('')
        print('  [OK] NEW FILE contains REAL ERA5 data')
        
        # Show sample values
        print('')
        print('Sample values (first frame, first channel, center 5x5):')
        center_h, center_w = f["past_frames"].shape[2] // 2, f["past_frames"].shape[3] // 2
        print(f["past_frames"][0, 0, center_h-2:center_h+3, center_w-2:center_w+3])
    else:
        print('')
        print('  [ERROR] NEW FILE still contains SYNTHETIC data (all zeros)')
else:
    print('No files found')











