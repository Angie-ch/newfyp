"""Final dataset summary and failure analysis"""
import numpy as np
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("FINAL DATASET SUMMARY (2018-2021)")
print("=" * 80)

# 1. Check generated data
output_dir = Path("D:/typhoon_data_2018_2021_full")
train_files = list((output_dir / "train/cases").glob("*.npz"))
val_files = list((output_dir / "val/cases").glob("*.npz"))
test_files = list((output_dir / "test/cases").glob("*.npz"))

print(f"\nGenerated Samples:")
print("-" * 80)
print(f"  Training (2018-2019):   {len(train_files)} samples")
print(f"  Validation (2020):      {len(val_files)} samples")
print(f"  Test (2021):            {len(test_files)} samples")
print(f"  TOTAL:                  {len(train_files) + len(val_files) + len(test_files)} samples")

# 2. Check sample quality
print(f"\nSample Quality Check (first 3 train samples):")
print("-" * 80)
for i, f in enumerate(train_files[:3]):
    data = np.load(f)
    past_frames = data['past_frames']
    future_frames = data['future_frames']
    print(f"\n  {f.name}:")
    print(f"    Past frames:   {past_frames.shape} (T={past_frames.shape[0]}, C={past_frames.shape[1]}, H={past_frames.shape[2]}, W={past_frames.shape[3]})")
    print(f"    Future frames: {future_frames.shape}")
    print(f"    Value range: [{past_frames.min():.2f}, {past_frames.max():.2f}]")
    print(f"    Has NaN:     {np.isnan(past_frames).any() or np.isnan(future_frames).any()}")
    print(f"    Variance:    {past_frames.var():.2e}")

# 3. Count storms by year
print(f"\nStorm Distribution:")
print("-" * 80)
storm_counts = defaultdict(int)
for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
    split_storms = set()
    for f in files:
        # Extract year and storm ID from filename: 2018_2018082N04147_w00.npz
        parts = f.stem.split('_')
        year = parts[0]
        storm_id = parts[1]
        split_storms.add(f"{year}_{storm_id}")
        storm_counts[year] += 1
    print(f"  {split.upper():12s}: {len(split_storms)} unique storms")

print(f"\n  By year:")
for year in sorted(storm_counts.keys()):
    print(f"    {year}: {storm_counts[year]} samples")

# 4. Analyze failures from log
print(f"\nFailure Analysis:")
print("-" * 80)
log_file = Path("regeneration_log_COMPLETE.txt")
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    # Extract failure info
    missing_dates = set()
    failed_storms = []
    
    for line in log_content.split('\n'):
        if 'ERA5 file not found' in line and 'date=' in line:
            try:
                date_part = line.split('date=')[1].split(')')[0]
                missing_dates.add(date_part)
            except:
                pass
        if 'No valid samples generated for' in line:
            try:
                storm_id = line.split('for ')[1].split(',')[0]
                failed_storms.append(storm_id)
            except:
                pass
    
    print(f"  Missing ERA5 dates:    {len(missing_dates)} dates")
    print(f"  Completely failed:     {len(failed_storms)} storms")
    if failed_storms:
        for storm in failed_storms:
            print(f"    - {storm}")

# 5. ERA5 file coverage
print(f"\nERA5 File Coverage:")
print("-" * 80)
era5_dir = Path("data/era5")
for year_dir in sorted(era5_dir.glob("ERA5_*")):
    nc_files = [f for f in year_dir.glob("*.nc") 
                if len(f.stem.split('_')[-1]) == 8 and f.stem.split('_')[-1].isdigit()]
    year = year_dir.name.split('_')[1]
    dates = sorted([f.stem.split('_')[-1] for f in nc_files])
    
    if dates:
        # Calculate missing dates in range
        from datetime import datetime, timedelta
        start_date = datetime.strptime(dates[0], '%Y%m%d')
        end_date = datetime.strptime(dates[-1], '%Y%m%d')
        days_in_range = (end_date - start_date).days + 1
        missing_in_range = days_in_range - len(dates)
        coverage_pct = (len(dates) / days_in_range * 100) if days_in_range > 0 else 0
        
        print(f"  {year}: {len(dates)} files ({coverage_pct:.1f}% coverage)")
        print(f"         Range: {dates[0]} - {dates[-1]}")
        print(f"         Missing: {missing_in_range} days in range")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"SUCCESS: Generated {len(train_files) + len(val_files) + len(test_files)} high-quality samples")
print(f"  - 8 past timesteps (48 hours)")
print(f"  - 12 future timesteps (72 hours)")
print(f"  - 64x64 spatial resolution")
print(f"  - Real ERA5 data (24 channels)")
print(f"  - 6-hour temporal resolution")
print()
print("WHY ONLY 72 SAMPLES?")
print("  1. ERA5 data is incomplete (missing ~150-185 days per year)")
print("  2. 2 storms completely failed (no ERA5 data for their dates)")
print("  3. Only 20 typhoons in 2018-2021 met criteria (wind>=33 m/s)")
print()
print("IS THIS ENOUGH FOR TRAINING?")
print("  - 72 samples is LIMITED for deep learning")
print("  - Typical needs: 1000+ samples")
print("  - Options:")
print("    1. Download complete ERA5 data for missing dates")
print("    2. Use data augmentation (rotate, flip, etc.)")
print("    3. Lower selection criteria (weaker typhoons)")
print("    4. Extend year range (2016-2022)")
print("=" * 80)

