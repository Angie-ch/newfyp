"""Monitor regeneration progress and verify ERA5 usage"""

import json
import time
from pathlib import Path

dataset_info_file = Path("data/processed_temporal_split/dataset_info.json")

print("="*80)
print("MONITORING REGENERATION - VERIFYING ERA5 USAGE")
print("="*80)
print("\nWaiting for regeneration to start...")
print("This will check the dataset_info.json file to verify ERA5 usage.\n")

last_modified = None
check_count = 0

while True:
    check_count += 1
    
    if dataset_info_file.exists():
        current_modified = dataset_info_file.stat().st_mtime
        
        if last_modified is None or current_modified > last_modified:
            last_modified = current_modified
            
            try:
                with open(dataset_info_file, 'r') as f:
                    info = json.load(f)
                
                print(f"\n[{time.strftime('%H:%M:%S')}] Dataset info updated!")
                print(f"  Meteorological data: {info.get('meteorological_data', 'Unknown')}")
                print(f"  Generation date: {info.get('generation_date', 'Unknown')}")
                print(f"  Total samples: {info.get('total_samples', 0)}")
                
                if info.get('meteorological_data') == 'ERA5':
                    print("\n  ✓✓✓ CONFIRMED: Using REAL ERA5 data! ✓✓✓")
                elif info.get('meteorological_data') == 'Synthetic':
                    print("\n  ⚠️  WARNING: Still showing Synthetic data!")
                    print("     Regeneration may not be complete yet.")
                else:
                    print(f"\n  Status: {info.get('meteorological_data', 'Unknown')}")
                
                # Check if regeneration is complete
                if info.get('total_samples', 0) > 0:
                    print(f"\n  Train samples: {info.get('splits', {}).get('train', {}).get('n_samples', 0)}")
                    print(f"  Val samples: {info.get('splits', {}).get('val', {}).get('n_samples', 0)}")
                    print(f"  Test samples: {info.get('splits', {}).get('test', {}).get('n_samples', 0)}")
                    
            except Exception as e:
                print(f"  Error reading file: {e}")
    else:
        if check_count == 1:
            print("  Dataset info file doesn't exist yet - regeneration starting...")
        elif check_count % 10 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] Still waiting for dataset_info.json...")
    
    time.sleep(5)  # Check every 5 seconds



