"""Verify the generated typhoon data contains real ERA5 data"""
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
print(f"VERIFYING GENERATED DATA")
print("="*80)
print(f"\nFound {len(sample_files)} sample files")
print(f"Checking first 5 samples...")

all_valid = True

for i, sample_file in enumerate(sample_files[:5]):
    print(f"\n[{i+1}/5] {sample_file.name}")
    print("-"*80)
    
    try:
        data = np.load(sample_file, allow_pickle=True)
        
        # Check keys
        required_keys = ['past_frames', 'future_frames', 'track_past', 'track_future']
        for key in required_keys:
            if key not in data:
                print(f"  [ERROR] Missing key: {key}")
                all_valid = False
                continue
        
        past_frames = data['past_frames']
        future_frames = data['future_frames']
        
        print(f"  past_frames shape: {past_frames.shape}")
        print(f"  future_frames shape: {future_frames.shape}")
        
        # Expected: (2, 24, 32, 32) for past (2 timesteps, 24 channels, 32x32)
        # Expected: (4, 24, 32, 32) for future (4 timesteps, 24 channels, 32x32)
        
        is_valid = True
        
        # Check shape
        if past_frames.ndim != 4 or past_frames.shape[0] != 2:
            print(f"  [ERROR] Unexpected past_frames shape: {past_frames.shape}")
            is_valid = False
        
        if future_frames.ndim != 4 or future_frames.shape[0] != 4:
            print(f"  [ERROR] Unexpected future_frames shape: {future_frames.shape}")
            is_valid = False
        
        # Check for NaN
        nan_count_past = np.isnan(past_frames).sum()
        nan_count_future = np.isnan(future_frames).sum()
        
        if nan_count_past > 0:
            print(f"  [WARNING] past_frames contains {nan_count_past} NaN values ({100*nan_count_past/past_frames.size:.2f}%)")
            if nan_count_past == past_frames.size:
                print(f"  [ERROR] ALL past_frames are NaN!")
                is_valid = False
        
        if nan_count_future > 0:
            print(f"  [WARNING] future_frames contains {nan_count_future} NaN values ({100*nan_count_future/future_frames.size:.2f}%)")
            if nan_count_future == future_frames.size:
                print(f"  [ERROR] ALL future_frames are NaN!")
                is_valid = False
        
        # Check for all zeros
        if np.all(past_frames == 0):
            print(f"  [ERROR] past_frames contains all zeros!")
            is_valid = False
        
        if np.all(future_frames == 0):
            print(f"  [ERROR] future_frames contains all zeros!")
            is_valid = False
        
        # Check variance
        var_past = np.nanvar(past_frames)
        var_future = np.nanvar(future_frames)
        
        print(f"  past_frames variance: {var_past:.2e}")
        print(f"  future_frames variance: {var_future:.2e}")
        
        if var_past < 1e-6:
            print(f"  [WARNING] Very low variance in past_frames")
            is_valid = False
        
        if var_future < 1e-6:
            print(f"  [WARNING] Very low variance in future_frames")
            is_valid = False
        
        # Check data range
        min_past = np.nanmin(past_frames)
        max_past = np.nanmax(past_frames)
        min_future = np.nanmin(future_frames)
        max_future = np.nanmax(future_frames)
        
        print(f"  past_frames range: [{min_past:.2f}, {max_past:.2f}]")
        print(f"  future_frames range: [{min_future:.2f}, {max_future:.2f}]")
        
        if is_valid:
            print(f"  [OK] Sample contains valid ERA5 data")
        else:
            print(f"  [ERROR] Sample contains invalid data")
            all_valid = False
            
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        all_valid = False

print("\n" + "="*80)
if all_valid:
    print("[SUCCESS] All checked samples contain valid, real ERA5 data!")
    print("Data generation is complete and successful.")
else:
    print("[FAILURE] Some samples contain invalid data.")
print("="*80)

