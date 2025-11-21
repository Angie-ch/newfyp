"""Find missing ERA5 dates needed for typhoon dataset"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

print("=" * 80)
print("FIND MISSING ERA5 DATES")
print("=" * 80)

# 1. Load typhoon tracks
print("\n[1] Loading typhoon tracks...")
tracks_file = Path("data/raw/temp_renamed_6h.csv")
df = pd.read_csv(tracks_file)

# 确保时间列存在
if 'ISO_TIME' in df.columns:
    df['datetime'] = pd.to_datetime(df['ISO_TIME'])
elif 'time' in df.columns:
    df['datetime'] = pd.to_datetime(df['time'])
else:
    print("[ERROR] No time column found!")
    exit(1)

# Filter 2018-2021 strong typhoons
df['year'] = df['datetime'].dt.year
df = df[(df['year'] >= 2018) & (df['year'] <= 2021)]

# Filter strong typhoons (wind>=33 m/s)
wind_col = 'WMO_WIND' if 'WMO_WIND' in df.columns else 'wind'
if wind_col in df.columns:
    df = df[df[wind_col] >= 33]

print(f"  Found {len(df)} strong typhoon records")

# 2. Extract all required dates
print("\n[2] Extracting required dates...")
required_dates = set()
for _, row in df.iterrows():
    dt = row['datetime']
    # Need +/-1 day buffer to ensure complete time window
    for offset in range(-1, 2):
        date = (dt + timedelta(days=offset)).date()
        required_dates.add(date)

print(f"  Total required dates: {len(required_dates)}")

# 3. Check existing ERA5 files
print("\n[3] Checking existing ERA5 files...")
era5_dir = Path("data/era5")
existing_dates = set()

for year_dir in era5_dir.glob("ERA5_*"):
    nc_files = list(year_dir.glob("era5_pl_*.nc"))
    for f in nc_files:
        try:
            date_str = f.stem.split('_')[-1]  # era5_pl_20180101.nc -> 20180101
            if len(date_str) == 8 and date_str.isdigit():
                date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                existing_dates.add(date_obj)
        except:
            pass

print(f"  Existing ERA5 files: {len(existing_dates)}")

# 4. Find missing dates
missing_dates = sorted(required_dates - existing_dates)
print(f"\n[4] Missing dates: {len(missing_dates)}")
print("=" * 80)

if missing_dates:
    # Group by year
    missing_by_year = defaultdict(list)
    for date in missing_dates:
        missing_by_year[date.year].append(date)
    
    print("\nBy year:")
    for year in sorted(missing_by_year.keys()):
        dates = missing_by_year[year]
        print(f"\n  {year}: {len(dates)} missing dates")
        print(f"    Date range: {dates[0]} to {dates[-1]}")
        if len(dates) <= 20:
            print(f"    Dates: {[str(d) for d in dates]}")
        else:
            print(f"    First 10: {[str(d) for d in dates[:10]]}")
            print(f"    Last 10: {[str(d) for d in dates[-10:]]}")
    
    # 5. Generate download list
    print("\n" + "=" * 80)
    print("Generating download list...")
    print("=" * 80)
    
    output_file = Path("missing_era5_dates.txt")
    with open(output_file, 'w') as f:
        f.write("# Missing ERA5 dates for typhoon dataset (2018-2021)\n")
        f.write(f"# Total: {len(missing_dates)} dates\n")
        f.write("# Format: YYYY-MM-DD\n\n")
        for year in sorted(missing_by_year.keys()):
            f.write(f"# {year} ({len(missing_by_year[year])} dates)\n")
            for date in missing_by_year[year]:
                f.write(f"{date}\n")
            f.write("\n")
    
    print(f"[OK] Missing dates list saved to: {output_file}")
    
    # 6. Estimate download size
    print("\n" + "=" * 80)
    print("Download Estimate:")
    print("=" * 80)
    print(f"  Missing dates: {len(missing_dates)} days")
    print(f"  File size per date: ~50-100 MB")
    print(f"  Estimated total download: ~{len(missing_dates) * 75 / 1024:.1f} GB")
    print(f"  Expected samples after download: ~100-150")
    
else:
    print("\n[OK] All required ERA5 files already exist!")

print("\n" + "=" * 80)

