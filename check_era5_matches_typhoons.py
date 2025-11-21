"""
Check if ERA5 files are cropped to match typhoon locations on specific dates
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr

print("="*80)
print("CHECKING IF ERA5 FILES MATCH TYPHOON LOCATIONS BY DATE")
print("="*80)
print()

# Load interpolated tracks directly
interpolated_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021.csv")
if not interpolated_file.exists():
    print(f"ERROR: {interpolated_file} not found")
    exit(1)

df = pd.read_csv(interpolated_file, low_memory=False)

# Get unique storms
storm_ids = df['typhoon_id'].unique()[:10]

matches = 0
mismatches = 0

for storm_id in storm_ids:
    storm_df = df[df['typhoon_id'] == storm_id].copy()
    if len(storm_df) == 0:
        continue
    
    # Get storm track
    lats = storm_df['lat'].values
    lons = storm_df['lon'].values
    times = pd.to_datetime(storm_df['time'].values)
    
    # Get date range
    dates = times.date
    unique_dates = sorted(set(dates))
    
    # Check a few dates
    for date in unique_dates[:3]:  # Check first 3 dates
        date_str = date.strftime('%Y%m%d')
        year = date.year
        
        # Get typhoon location on this date
        date_mask = dates == date
        if not date_mask.any():
            continue
        
        typhoon_lats = lats[date_mask]
        typhoon_lons = lons[date_mask]
        typhoon_lat = float(np.mean(typhoon_lats))
        typhoon_lon = float(np.mean(typhoon_lons))
        
        # Check if ERA5 file exists for this date
        year_dir = Path(f"data/era5/ERA5_{year}_26data")
        era5_file = year_dir / f"era5_pl_{date_str}.nc"
        
        if not era5_file.exists():
            continue
        
        # Get ERA5 file coverage
        try:
            ds = xr.open_dataset(era5_file)
            file_lon_min = float(ds.coords['longitude'].min().values)
            file_lon_max = float(ds.coords['longitude'].max().values)
            file_lat_min = float(ds.coords['latitude'].min().values)
            file_lat_max = float(ds.coords['latitude'].max().values)
            ds.close()
            
            # Check if typhoon is within file coverage
            in_coverage = (
                file_lon_min <= typhoon_lon <= file_lon_max and
                file_lat_min <= typhoon_lat <= file_lat_max
            )
            
            if in_coverage:
                matches += 1
                status = "[MATCH]"
            else:
                mismatches += 1
                status = "[MISMATCH]"
            
            print(f"{date_str} - Storm {storm_id[:10]}:")
            print(f"  Typhoon: ({typhoon_lat:.2f}°N, {typhoon_lon:.2f}°E)")
            print(f"  File: lat=[{file_lat_min:.2f}, {file_lat_max:.2f}], lon=[{file_lon_min:.2f}, {file_lon_max:.2f}]")
            print(f"  {status}")
            print()
        
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

print("="*80)
print(f"SUMMARY: {matches} matches, {mismatches} mismatches")
if matches > 0 and mismatches == 0:
    print("[RESULT] YES - Files appear to be cropped to match typhoon locations!")
elif matches > mismatches:
    print("[RESULT] MOSTLY - Most files match typhoon locations")
else:
    print("[RESULT] NO - Files don't consistently match typhoon locations")










