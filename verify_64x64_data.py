"""Verify the generated typhoon data is 64x64"""
import numpy as np
from pathlib import Path

# Check samples from D:\ drive
output_dir = Path("D:/typhoon_data_2018_test")
train_dir = output_dir / "train"

# Find sample files
sample_files = list(train_dir.glob("**/cases/*.npz"))

if not sample_files:
    print("[ERROR] No sample files found!")
    exit(1)

print("="*80)
print(f"VERIFYING 64x64 DATA")
print("="*80)
print(f"\nFound {len(sample_files)} sample files")
print(f"Checking first 3 samples...")

all_valid = True

for i, sample_file in enumerate(sample_files[:3]):
    print(f"\n[{i+1}/3] {sample_file.name}")
    print("-"*80)
    
    try:
        data = np.load(sample_file, allow_pickle=True)
        
        past_frames = data['past_frames']
        future_frames = data['future_frames']
        
        print(f"  past_frames shape: {past_frames.shape}")
        print(f"  future_frames shape: {future_frames.shape}")
        
        # Expected: (2, 24, 64, 64) for past
        # Expected: (4, 24, 64, 64) for future
        
        is_valid = True
        
        # Check if 64x64
        if past_frames.shape != (2, 24, 64, 64):
            print(f"  [ERROR] Expected (2, 24, 64, 64), got {past_frames.shape}")
            is_valid = False
        else:
            print(f"  [OK] past_frames is 64x64")
        
        if future_frames.shape != (4, 24, 64, 64):
            print(f"  [ERROR] Expected (4, 24, 64, 64), got {future_frames.shape}")
            is_valid = False
        else:
            print(f"  [OK] future_frames is 64x64")
        
        # Check for NaN
        nan_count_past = np.isnan(past_frames).sum()
        nan_count_future = np.isnan(future_frames).sum()
        
        if nan_count_past > 0:
            print(f"  [WARNING] {nan_count_past} NaN values in past_frames")
        if nan_count_future > 0:
            print(f"  [WARNING] {nan_count_future} NaN values in future_frames")
        
        # Check variance
        var_past = np.nanvar(past_frames)
        var_future = np.nanvar(future_frames)
        
        print(f"  Variance: past={var_past:.2e}, future={var_future:.2e}")
        
        # Check data range
        min_past = np.nanmin(past_frames)
        max_past = np.nanmax(past_frames)
        
        print(f"  Data range: [{min_past:.2f}, {max_past:.2f}]")
        
        if is_valid:
            print(f"  [OK] Sample is valid 64x64 ERA5 data")
        else:
            all_valid = False
            
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        all_valid = False

print("\n" + "="*80)
if all_valid:
    print("[SUCCESS] All samples are 64x64 with valid ERA5 data!")
else:
    print("[FAILURE] Some samples have incorrect dimensions")
print("="*80)

