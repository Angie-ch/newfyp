"""
Regenerate dataset with real ERA5 data and monitor progress
"""
import subprocess
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

def check_sample_quality(sample_file):
    """Check if a sample file has real data (not all zeros)"""
    try:
        data = np.load(sample_file, allow_pickle=True)
        if 'past_frames' in data:
            frames = data['past_frames']
            non_zero_pct = (frames != 0).sum() / frames.size * 100
            has_data = non_zero_pct > 1.0  # At least 1% non-zero
            return has_data, non_zero_pct, frames.min(), frames.max(), frames.mean(), frames.std()
        return False, 0, 0, 0, 0, 0
    except Exception as e:
        return False, 0, 0, 0, 0, 0

def monitor_regeneration():
    """Monitor regeneration progress"""
    print("="*80)
    print("MONITORING DATASET REGENERATION")
    print("="*80)
    print()
    
    output_dir = Path("data/processed_temporal_split")
    train_dir = output_dir / "train" / "cases"
    val_dir = output_dir / "val" / "cases"
    test_dir = output_dir / "test" / "cases"
    
    last_train_count = 0
    last_val_count = 0
    last_test_count = 0
    samples_checked = set()
    
    print("Monitoring for new samples...")
    print("Checking sample quality (non-zero data)...")
    print()
    
    try:
        while True:
            train_count = len(list(train_dir.glob("*.npz"))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob("*.npz"))) if val_dir.exists() else 0
            test_count = len(list(test_dir.glob("*.npz"))) if test_dir.exists() else 0
            
            # Check new samples for quality
            if train_count > last_train_count:
                new_samples = [f for f in train_dir.glob("*.npz") if f.name not in samples_checked]
                for sample_file in new_samples[:5]:  # Check first 5 new samples
                    has_data, non_zero, min_val, max_val, mean_val, std_val = check_sample_quality(sample_file)
                    samples_checked.add(sample_file.name)
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if has_data:
                        print(f"[{timestamp}] ✓ {sample_file.name}: {non_zero:.1f}% non-zero, range=[{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}, std={std_val:.3f}")
                    else:
                        print(f"[{timestamp}] ✗ {sample_file.name}: ALL ZEROS (no data!)" + " " * 20 + "[WARNING]")
            
            if train_count != last_train_count or val_count != last_val_count or test_count != last_test_count:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Progress: Train={train_count}, Val={val_count}, Test={test_count}, Total={train_count + val_count + test_count}")
                
                last_train_count = train_count
                last_val_count = val_count
                last_test_count = test_count
            
            # Check if dataset_info.json exists and was updated
            info_file = output_dir / "dataset_info.json"
            if info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    if info.get('meteorological_data') == 'ERA5':
                        print()
                        print("="*80)
                        print("✓ REGENERATION COMPLETE WITH REAL ERA5 DATA!")
                        print("="*80)
                        print(f"  Meteorological data: {info['meteorological_data']}")
                        print(f"  Total samples: {info.get('total_samples', 'N/A')}")
                        
                        # Final quality check
                        print()
                        print("Performing final quality check on samples...")
                        sample_files = list(train_dir.glob("*.npz"))[:10]  # Check 10 samples
                        good_samples = 0
                        bad_samples = 0
                        for sample_file in sample_files:
                            has_data, non_zero, _, _, _, _ = check_sample_quality(sample_file)
                            if has_data:
                                good_samples += 1
                            else:
                                bad_samples += 1
                        
                        print(f"  Quality check: {good_samples}/{len(sample_files)} samples have real data")
                        if bad_samples > 0:
                            print(f"  ⚠ WARNING: {bad_samples} samples are empty (all zeros)")
                        else:
                            print(f"  ✓ All checked samples contain real ERA5 data!")
                        
                        return True
                except:
                    pass
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        return False

def main():
    """Main function"""
    print("="*80)
    print("REGENERATING DATASET WITH REAL ERA5 DATA")
    print("="*80)
    print()
    
    # Use virtual environment Python if available
    venv_python = Path("pytorch_gpu/Scripts/python.exe")
    if venv_python.exists():
        python_cmd = str(venv_python)
        print(f"Using virtual environment Python: {python_cmd}")
    else:
        python_cmd = sys.executable
        print(f"Using system Python: {python_cmd}")
    
    print()
    print("Starting regeneration process...")
    print("This will:")
    print("  1. Load real ERA5 data from data/era5/")
    print("  2. Extract ERA5 frames for each typhoon timestep")
    print("  3. Create new .npz files with real data")
    print("  4. Update dataset_info.json to show ERA5")
    print()
    print("Monitoring will check:")
    print("  - Sample count progress")
    print("  - Sample quality (non-zero data)")
    print("  - ERA5 data extraction success")
    print()
    print("="*80)
    print()
    
    # Start regeneration in background
    cmd = [python_cmd, "data/generate_data_by_year.py"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start monitoring in a separate thread/process
        import threading
        monitor_thread = threading.Thread(target=monitor_regeneration, daemon=True)
        monitor_thread.start()
        
        # Also print process output
        print("Regeneration process output:")
        print("-"*80)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        # Wait a bit for monitoring to catch up
        time.sleep(5)
        
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n\nRegeneration interrupted by user.")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())










