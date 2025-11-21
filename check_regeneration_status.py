"""
Quick check of regeneration status
"""
import json
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("REGENERATION STATUS CHECK")
    print("="*80)
    print()
    
    # Check dataset info
    info_file = Path("data/processed_temporal_split/dataset_info.json")
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print("Dataset Metadata:")
        print(f"  Meteorological Data: {info.get('meteorological_data', 'Unknown')}")
        if info.get('meteorological_data') == 'ERA5':
            print("  [OK] Using REAL ERA5 data!" + " " * 30 + "[PASS]")
        else:
            print("  [FAIL] Not using real ERA5 data" + " " * 25 + "[FAIL]")
        print(f"  Total Samples: {info.get('total_samples', 'N/A')}")
        print(f"  Generation Date: {info.get('generation_date', 'N/A')}")
        print()
    else:
        print("Dataset info file not found - regeneration may still be in progress")
        print()
    
    # Check sample quality
    train_dir = Path("data/processed_temporal_split/train/cases")
    if train_dir.exists():
        # Filter out macOS resource fork files
        npz_files = [f for f in train_dir.glob("*.npz") if not f.name.startswith('._')]
        print(f"Sample Files: {len(npz_files)} found (excluding system files)")
        
        if npz_files:
            # Check a few samples
            print()
            print("Checking sample quality (first 5 samples):")
            good = 0
            bad = 0
            
            for sample_file in sorted(npz_files)[:5]:
                try:
                    data = np.load(sample_file, allow_pickle=True)
                    if 'past_frames' in data:
                        frames = data['past_frames']
                        non_zero = (frames != 0).sum() / frames.size * 100
                        if non_zero > 1.0:
                            print(f"  [OK] {sample_file.name}: {non_zero:.1f}% non-zero, range=[{frames.min():.3f}, {frames.max():.3f}]")
                            good += 1
                        else:
                            print(f"  [BAD] {sample_file.name}: ALL ZEROS" + " " * 30 + "[BAD]")
                            bad += 1
                except Exception as e:
                    print(f"  [ERROR] {sample_file.name}: Error - {e}")
                    bad += 1
            
            print()
            print(f"Quality Summary: {good} good, {bad} bad")
            
            if bad > 0:
                print("[WARNING] Some samples are empty (all zeros)")
                print("  Regeneration may have failed or is still in progress")
            elif good > 0:
                print("[OK] Samples contain real data")
        else:
            print("  No sample files found yet")
    
    print()
    print("="*80)

if __name__ == "__main__":
    main()

