"""
Download missing ERA5 data using CDS API
Requires: cdsapi package and CDS API credentials

Setup instructions:
1. Install cdsapi: pip install cdsapi
2. Register at: https://cds.climate.copernicus.eu/
3. Get API key from: https://cds.climate.copernicus.eu/api-how-to
4. Create ~/.cdsapirc with:
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY
"""

import cdsapi
from pathlib import Path
from datetime import datetime
import time

print("=" * 80)
print("DOWNLOAD MISSING ERA5 DATA")
print("=" * 80)

# Read missing dates
missing_dates_file = Path("missing_era5_dates.txt")
if not missing_dates_file.exists():
    print("[ERROR] missing_era5_dates.txt not found!")
    print("Please run: python find_missing_era5_dates.py first")
    exit(1)

# Parse missing dates
missing_dates = []
with open(missing_dates_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                date = datetime.strptime(line, '%Y-%m-%d')
                missing_dates.append(date)
            except:
                pass

print(f"\n[INFO] Found {len(missing_dates)} missing dates to download")

# ERA5 pressure level variables (matching your existing data)
variables = [
    'geopotential',      # z
    'temperature',       # t
    'u_component_of_wind',  # u
    'v_component_of_wind',  # v
    'relative_humidity',    # r
    'vertical_velocity',    # w (or vo)
]

pressure_levels = [
    '200', '300', '500', '700', '850', '925'
]

# Initialize CDS API client
try:
    c = cdsapi.Client()
    print("[OK] CDS API client initialized")
except Exception as e:
    print(f"[ERROR] Failed to initialize CDS API: {e}")
    print("\nPlease check:")
    print("1. Install cdsapi: pip install cdsapi")
    print("2. Setup ~/.cdsapirc with your API credentials")
    print("3. Get credentials from: https://cds.climate.copernicus.eu/api-how-to")
    exit(1)

# Download each missing date
print(f"\n[INFO] Starting download...")
print("=" * 80)

era5_base_dir = Path("data/era5")
successful = 0
failed = []

for i, date in enumerate(missing_dates, 1):
    year = date.year
    month = f"{date.month:02d}"
    day = f"{date.day:02d}"
    date_str = date.strftime('%Y%m%d')
    
    # Determine output directory and filename
    year_dir = era5_base_dir / f"ERA5_{year}_26data"
    year_dir.mkdir(parents=True, exist_ok=True)
    output_file = year_dir / f"era5_pl_{date_str}.nc"
    
    if output_file.exists():
        print(f"[{i}/{len(missing_dates)}] {date_str} - Already exists, skipping")
        successful += 1
        continue
    
    print(f"[{i}/{len(missing_dates)}] Downloading {date_str}...")
    
    try:
        # Download request
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'pressure_level': pressure_levels,
                'year': str(year),
                'month': month,
                'day': day,
                'time': [
                    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
                ],
                'area': [60, 90, -10, 180],  # N, W, S, E - covers Western Pacific
            },
            str(output_file)
        )
        print(f"  [OK] Downloaded: {output_file}")
        successful += 1
        
        # Add small delay to avoid overwhelming the server
        time.sleep(2)
        
    except Exception as e:
        print(f"  [ERROR] Failed to download {date_str}: {e}")
        failed.append(date_str)
        if output_file.exists():
            output_file.unlink()  # Remove incomplete file
        continue

# Summary
print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"  Total dates:  {len(missing_dates)}")
print(f"  Successful:   {successful}")
print(f"  Failed:       {len(failed)}")

if failed:
    print(f"\nFailed dates:")
    for date in failed:
        print(f"  - {date}")
    
    # Save failed dates for retry
    failed_file = Path("failed_era5_downloads.txt")
    with open(failed_file, 'w') as f:
        for date in failed:
            f.write(f"{date}\n")
    print(f"\n[INFO] Failed dates saved to: {failed_file}")

if successful > 0:
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Run: python data/generate_data_by_year.py")
    print("2. Expected samples: ~100-150 (up from 72)")
    print("=" * 80)

