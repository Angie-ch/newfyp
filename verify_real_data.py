"""
Verify if the dataset contains real ERA5 data
"""
import json
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("VERIFYING REAL ERA5 DATA IN DATASET")
    print("="*80)
    print()
    
    # Check 1: Dataset info
    info_file = Path("data/processed_temporal_split/dataset_info.json")
    if not info_file.exists():
        print("❌ ERROR: dataset_info.json not found!")
        print("   Dataset may not have been generated yet.")
        return False
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    print("1. Dataset Metadata:")
    print(f"   Meteorological Data: {info.get('meteorological_data', 'Unknown')}")
    
    if info.get('meteorological_data') == 'ERA5':
        print("   ✓ Using REAL ERA5 data!" + " " * 30 + "[PASS]")
    elif info.get('meteorological_data') == 'Synthetic':
        print("   ✗ Using SYNTHETIC data (not real ERA5)" + " " * 10 + "[FAIL]")
        return False
    else:
        print("   ⚠ Unknown data type" + " " * 30 + "[UNKNOWN]")
    
    print(f"   Total Samples: {info.get('total_samples', 'N/A')}")
    print(f"   Generation Date: {info.get('generation_date', 'N/A')}")
    print()
    
    # Check 2: Sample file inspection
    print("2. Inspecting Sample Files:")
    train_dir = Path("data/processed_temporal_split/train/cases")
    
    if not train_dir.exists():
        print("   ✗ Training cases directory not found!" + " " * 20 + "[FAIL]")
        return False
    
    npz_files = list(train_dir.glob("*.npz"))
    if not npz_files:
        print("   ✗ No .npz files found!" + " " * 30 + "[FAIL]")
        return False
    
    print(f"   Found {len(npz_files)} training samples")
    
    # Load a sample file and check its contents
    sample_file = npz_files[0]
    print(f"   Inspecting: {sample_file.name}")
    
    try:
        data = np.load(sample_file, allow_pickle=True)
        
        # Check for past_frames
        if 'past_frames' in data:
            past_frames = data['past_frames']
            print(f"   Past frames shape: {past_frames.shape}")
            
            # Check if data is all zeros (would indicate problem)
            if (past_frames == 0).all():
                print("   ✗ All frames are zeros!" + " " * 30 + "[FAIL]")
                return False
            else:
                non_zero_pct = (past_frames != 0).sum() / past_frames.size * 100
                print(f"   Non-zero data: {non_zero_pct:.2f}%")
                
                # Check data range (real ERA5 should have reasonable values)
                print(f"   Data range: [{past_frames.min():.4f}, {past_frames.max():.4f}]")
                print(f"   Data mean: {past_frames.mean():.4f}")
                print(f"   Data std: {past_frames.std():.4f}")
                
                # Real ERA5 data typically has:
                # - Non-zero values
                # - Reasonable range (not all same value)
                # - Some variation
                if non_zero_pct > 10 and past_frames.std() > 0.1:
                    print("   ✓ Data looks like real ERA5 values" + " " * 15 + "[PASS]")
                else:
                    print("   ⚠ Data may be synthetic or corrupted" + " " * 10 + "[WARNING]")
        else:
            print("   ✗ No 'past_frames' found in sample!" + " " * 20 + "[FAIL]")
            return False
        
        # Check for future_frames
        if 'future_frames' in data:
            future_frames = data['future_frames']
            print(f"   Future frames shape: {future_frames.shape}")
        
        # Check metadata
        if 'storm_id' in data:
            print(f"   Storm ID: {data['storm_id']}")
        if 'storm_name' in data:
            print(f"   Storm Name: {data['storm_name']}")
        if 'year' in data:
            print(f"   Year: {data['year']}")
            
    except Exception as e:
        print(f"   ✗ Error loading sample: {e}" + " " * 20 + "[FAIL]")
        return False
    
    print()
    
    # Check 3: Count samples
    print("3. Sample Counts:")
    train_count = len(list(train_dir.glob("*.npz")))
    val_count = len(list(Path("data/processed_temporal_split/val/cases").glob("*.npz"))) if Path("data/processed_temporal_split/val/cases").exists() else 0
    test_count = len(list(Path("data/processed_temporal_split/test/cases").glob("*.npz"))) if Path("data/processed_temporal_split/test/cases").exists() else 0
    
    print(f"   Train: {train_count}")
    print(f"   Val: {val_count}")
    print(f"   Test: {test_count}")
    print(f"   Total: {train_count + val_count + test_count}")
    print()
    
    # Final verdict
    print("="*80)
    if info.get('meteorological_data') == 'ERA5':
        print("✓ VERIFICATION PASSED: Dataset uses REAL ERA5 data!")
        print("="*80)
        return True
    else:
        print("✗ VERIFICATION FAILED: Dataset does NOT use real ERA5 data")
        print("="*80)
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)











