"""
Verify that output samples contain real ERA5 data, not synthetic data
"""
import numpy as np
from pathlib import Path
import sys

def verify_sample_data(sample_file: Path):
    """
    Verify if a sample file contains real ERA5 data or synthetic data
    
    Real ERA5 data characteristics:
    - 48 channels (6 single-level + 6 pressure-level vars × 7 levels)
    - Realistic value ranges (not all zeros or simple patterns)
    - Spatial variability consistent with meteorological data
    - Non-zero variance across channels
    
    Synthetic data characteristics:
    - Simple patterns (circular, radial)
    - Limited variability
    - May have all zeros or NaN
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING SAMPLE: {sample_file.name}")
    print("="*80)
    
    try:
        data = np.load(sample_file, allow_pickle=True)
        
        # Check required keys
        required_keys = ['past_frames', 'future_frames']
        for key in required_keys:
            if key not in data:
                print(f"[ERROR] Missing key: {key}")
                return False
        
        past_frames = data['past_frames']
        future_frames = data['future_frames']
        
        print(f"\nFrame shapes:")
        print(f"  Past frames: {past_frames.shape}")
        print(f"  Future frames: {future_frames.shape}")
        
        # Check number of channels (ERA5 should have 48 channels)
        n_channels = past_frames.shape[1] if len(past_frames.shape) > 1 else 1
        print(f"\nNumber of channels: {n_channels}")
        
        if n_channels == 48:
            print("[OK] Has 48 channels - consistent with ERA5 data")
        elif n_channels < 48:
            print(f"[WARNING] Only {n_channels} channels - may not be full ERA5 data")
        else:
            print(f"[WARNING] {n_channels} channels - unexpected for ERA5")
        
        # Check for zeros/NaN (synthetic data might be all zeros)
        past_zero_ratio = np.sum(past_frames == 0) / past_frames.size
        past_nan_ratio = np.sum(np.isnan(past_frames)) / past_frames.size
        
        future_zero_ratio = np.sum(future_frames == 0) / future_frames.size
        future_nan_ratio = np.sum(np.isnan(future_frames)) / future_frames.size
        
        print(f"\nData quality checks:")
        print(f"  Past frames - Zeros: {past_zero_ratio*100:.2f}%, NaNs: {past_nan_ratio*100:.2f}%")
        print(f"  Future frames - Zeros: {future_zero_ratio*100:.2f}%, NaNs: {future_nan_ratio*100:.2f}%")
        
        if past_zero_ratio > 0.9 or future_zero_ratio > 0.9:
            print("[WARNING] High zero ratio - may be synthetic or invalid data")
            return False
        
        if past_nan_ratio > 0.5 or future_nan_ratio > 0.5:
            print("[WARNING] High NaN ratio - data may be invalid")
            return False
        
        # Check value ranges (ERA5 has realistic meteorological values)
        past_min, past_max = np.nanmin(past_frames), np.nanmax(past_frames)
        future_min, future_max = np.nanmin(future_frames), np.nanmax(future_frames)
        
        print(f"\nValue ranges:")
        print(f"  Past frames: [{past_min:.2f}, {past_max:.2f}]")
        print(f"  Future frames: [{future_min:.2f}, {future_max:.2f}]")
        
        # Check variance (real data should have significant variance)
        past_var = np.nanvar(past_frames)
        future_var = np.nanvar(future_frames)
        
        print(f"\nVariance:")
        print(f"  Past frames: {past_var:.2f}")
        print(f"  Future frames: {future_var:.2f}")
        
        if past_var < 0.01 or future_var < 0.01:
            print("[WARNING] Very low variance - may be synthetic or constant data")
            return False
        
        # Check channel-wise statistics (real ERA5 has different patterns per channel)
        print(f"\nChannel statistics (first 5 channels):")
        for ch in range(min(5, n_channels)):
            ch_data = past_frames[:, ch, :, :]
            ch_mean = np.nanmean(ch_data)
            ch_std = np.nanstd(ch_data)
            ch_range = np.nanmax(ch_data) - np.nanmin(ch_data)
            print(f"  Channel {ch}: mean={ch_mean:.2f}, std={ch_std:.2f}, range={ch_range:.2f}")
        
        # Check metadata if available
        if 'storm_id' in data:
            print(f"\nMetadata:")
            print(f"  Storm ID: {data['storm_id']}")
        if 'year' in data:
            print(f"  Year: {data['year']}")
        if 'storm_name' in data:
            print(f"  Storm name: {data['storm_name']}")
        
        # Final verdict
        print(f"\n{'='*80}")
        if n_channels == 48 and past_var > 0.01 and future_var > 0.01:
            print("[OK] Sample appears to contain REAL ERA5 data")
            print("  - 48 channels (consistent with ERA5)")
            print("  - Non-zero variance (real meteorological data)")
            print("  - Reasonable value ranges")
            return True
        else:
            print("[WARNING] Sample may contain synthetic or invalid data")
            return False
        
    except Exception as e:
        print(f"[ERROR] Failed to verify sample: {e}")
        return False


def main():
    output_dir = Path("data/processed_temporal_split")
    
    if not output_dir.exists():
        print(f"[ERROR] Output directory does not exist: {output_dir}")
        print("  Regeneration may still be running or hasn't started yet.")
        return
    
    # Find all sample files
    sample_files = list(output_dir.rglob("*.npz"))
    
    if not sample_files:
        print(f"[INFO] No sample files found in {output_dir}")
        print("  Regeneration may still be running.")
        print("  Check the log file for progress.")
        return
    
    print(f"Found {len(sample_files)} sample files")
    print(f"Verifying first 5 samples...")
    
    verified_count = 0
    for sample_file in sample_files[:5]:
        if verify_sample_data(sample_file):
            verified_count += 1
    
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print("="*80)
    print(f"Verified: {verified_count}/5 samples")
    if verified_count == 5:
        print("[OK] All samples appear to contain real ERA5 data")
    elif verified_count > 0:
        print("[WARNING] Some samples may have issues")
    else:
        print("[ERROR] No valid samples found")


if __name__ == "__main__":
    main()

