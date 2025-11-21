"""
Monitor dataset regeneration progress
"""
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_progress():
    """Check regeneration progress"""
    print("="*80)
    print("MONITORING DATASET REGENERATION PROGRESS")
    print("="*80)
    print()
    
    output_dir = Path("data/processed_temporal_split")
    train_dir = output_dir / "train" / "cases"
    val_dir = output_dir / "val" / "cases"
    test_dir = output_dir / "test" / "cases"
    
    print("Checking for generated samples...")
    print()
    
    last_train_count = 0
    last_val_count = 0
    last_test_count = 0
    
    try:
        while True:
            train_count = len(list(train_dir.glob("*.npz"))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob("*.npz"))) if val_dir.exists() else 0
            test_count = len(list(test_dir.glob("*.npz"))) if test_dir.exists() else 0
            
            if train_count != last_train_count or val_count != last_val_count or test_count != last_test_count:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Progress Update:")
                print(f"  Train samples: {train_count}")
                print(f"  Val samples: {val_count}")
                print(f"  Test samples: {test_count}")
                print(f"  Total: {train_count + val_count + test_count}")
                print()
                
                last_train_count = train_count
                last_val_count = val_count
                last_test_count = test_count
            
            # Check if dataset_info.json exists and was updated
            info_file = output_dir / "dataset_info.json"
            if info_file.exists():
                import json
                try:
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    if info.get('meteorological_data') == 'ERA5':
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Dataset regeneration complete!")
                        print(f"  Meteorological data: {info['meteorological_data']}")
                        print(f"  Total samples: {info.get('total_samples', 'N/A')}")
                        print()
                        print("Ready to start training!")
                        break
                except:
                    pass
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    check_progress()











