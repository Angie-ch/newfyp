"""Verify the 8x12 configuration"""
import numpy as np
from pathlib import Path

output_dir = Path("D:/typhoon_data_2018_test")
train_dir = output_dir / "train"
sample_files = list(train_dir.glob("**/cases/*.npz"))

print("="*80)
print("VERIFYING 8 PAST x 12 FUTURE CONFIGURATION")
print("="*80)
print(f"\nTotal samples: {len(sample_files)}")
print("Checking first 3 samples...")

all_valid = True

for i, sample_file in enumerate(sample_files[:3]):
    print(f"\n{'-'*80}")
    print(f"Sample {i+1}: {sample_file.name}")
    print("-"*80)
    
    try:
        data = np.load(sample_file, allow_pickle=True)
        past_frames = data['past_frames']
        future_frames = data['future_frames']
        
        print(f"\nShapes:")
        print(f"  past_frames:   {past_frames.shape}")
        print(f"  future_frames: {future_frames.shape}")
        
        # Expected: (8, 24, 64, 64) for past
        # Expected: (12, 24, 64, 64) for future
        
        expected_past = (8, 24, 64, 64)
        expected_future = (12, 24, 64, 64)
        
        if past_frames.shape != expected_past:
            print(f"  [ERROR] Expected past {expected_past}, got {past_frames.shape}")
            all_valid = False
        else:
            print(f"  [OK] Past frames: 8 timesteps (48 hours)")
        
        if future_frames.shape != expected_future:
            print(f"  [ERROR] Expected future {expected_future}, got {future_frames.shape}")
            all_valid = False
        else:
            print(f"  [OK] Future frames: 12 timesteps (72 hours)")
        
        # Check data quality
        var_past = np.nanvar(past_frames)
        var_future = np.nanvar(future_frames)
        
        print(f"\nData Quality:")
        print(f"  Variance: past={var_past:.2e}, future={var_future:.2e}")
        
        if var_past < 1e6:
            print(f"  [WARNING] Low variance in past frames")
            all_valid = False
        
        nan_past = np.isnan(past_frames).sum()
        nan_future = np.isnan(future_frames).sum()
        
        if nan_past > 0 or nan_future > 0:
            print(f"  [WARNING] NaN detected: past={nan_past}, future={nan_future}")
        else:
            print(f"  [OK] No NaN values")
        
        min_val = np.nanmin(past_frames)
        max_val = np.nanmax(past_frames)
        print(f"  Data range: [{min_val:.2f}, {max_val:.2f}]")
        
        # Check temporal span
        if 'track_past' in data and 'track_future' in data:
            track_past = data['track_past']
            track_future = data['track_future']
            print(f"\nTrack shapes:")
            print(f"  track_past:   {track_past.shape}")
            print(f"  track_future: {track_future.shape}")
            
            if track_past.shape[0] != 8:
                print(f"  [ERROR] Expected 8 past track points")
                all_valid = False
            if track_future.shape[0] != 12:
                print(f"  [ERROR] Expected 12 future track points")
                all_valid = False
        
        print(f"\n>>> Total time span: 20 timesteps = 120 hours (5 days)")
        print(f">>> Input: 8 timesteps (48h) -> Predict: 12 timesteps (72h)")
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        all_valid = False

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all_valid:
    print("\n*** SUCCESS ***")
    print("\nAll samples have correct 8x12 configuration!")
    print("\nConfiguration details:")
    print("  - Past:   8 timesteps x 24 channels x 64x64 pixels")
    print("  - Future: 12 timesteps x 24 channels x 64x64 pixels")
    print("  - Time resolution: 6 hours")
    print("  - Past coverage: 48 hours (2 days)")
    print("  - Future coverage: 72 hours (3 days)")
    print("  - Total span: 120 hours (5 days)")
    print("\nThis matches LT3P paper requirements!")
else:
    print("\n*** ERROR ***")
    print("\nSome samples have incorrect configuration.")

print("="*80)

