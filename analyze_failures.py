"""Analyze data generation failures"""
import pandas as pd
from pathlib import Path
from collections import defaultdict
import os

print("=" * 80)
print("DATA GENERATION FAILURE ANALYSIS")
print("=" * 80)

# 1. 读取日志文件
log_file = Path("regeneration_log_COMPLETE.txt")
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    log_content = f.read()

# 2. 统计缺失的ERA5文件
missing_files = defaultdict(list)
for line in log_content.split('\n'):
    if 'ERA5 file not found' in line and 'date=' in line:
        # 提取日期
        date_part = line.split('date=')[1].split(')')[0]
        storm_id = line.split('for ')[1].split(',')[0]
        missing_files[date_part].append(storm_id)

print(f"\nMissing ERA5 files ({len(missing_files)} dates):")
print("-" * 80)
for date, storms in sorted(missing_files.items()):
    print(f"  {date}: {len(set(storms))} storms affected")

# 3. Count completely failed storms
failed_storms = []
for line in log_content.split('\n'):
    if 'No valid samples generated for' in line:
        storm_id = line.split('for ')[1].split(',')[0]
        failed_storms.append(storm_id)

print(f"\nCompletely failed storms ({len(failed_storms)}):")
print("-" * 80)
for storm in failed_storms:
    print(f"  {storm}")

# 4. Check ERA5 file coverage
print(f"\nERA5 file coverage:")
print("-" * 80)
era5_dir = Path("data/era5")
for year_dir in sorted(era5_dir.glob("ERA5_*")):
    nc_files = list(year_dir.glob("*.nc"))
    year = year_dir.name.split('_')[1]
    print(f"  {year}: {len(nc_files)} files")
    
    # Check date continuity
    dates = []
    for f in nc_files:
        try:
            date_str = f.stem.split('_')[-1]  # era5_pl_20180101.nc -> 20180101
            # Only include 8-digit dates
            if len(date_str) == 8 and date_str.isdigit():
                dates.append(date_str)
        except:
            pass
    
    dates = sorted(dates)
    if dates:
        print(f"    Date range: {dates[0]} - {dates[-1]}")
        
        # Find missing dates
        from datetime import datetime, timedelta
        start_date = datetime.strptime(dates[0], '%Y%m%d')
        end_date = datetime.strptime(dates[-1], '%Y%m%d')
        expected_dates = set()
        current = start_date
        while current <= end_date:
            expected_dates.add(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
        
        actual_dates = set(dates)
        missing = expected_dates - actual_dates
        if missing:
            print(f"    Missing dates: {len(missing)}")
            if len(missing) <= 20:
                print(f"    Missing: {sorted(list(missing))[:10]}...")

# 5. Final results
print(f"\nFinal generation results:")
print("-" * 80)
output_dir = Path("D:/typhoon_data_2018_2021_full")
if output_dir.exists():
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            npz_files = list(split_dir.glob("*.npz"))
            print(f"  {split}: {len(npz_files)} samples")

print("\n" + "=" * 80)
print("ANALYSIS & RECOMMENDATIONS:")
print("=" * 80)
print("1. Incomplete ERA5 files are the main cause")
print("2. Missing ERA5 dates caused typhoon sample generation failures")
print("3. Recommendations:")
print("   - Check ERA5 data source and fill missing dates")
print("   - OR accept current 72 samples for model training")
print("   - 72 samples may not be enough for deep learning (typically need hundreds)")
print("=" * 80)

