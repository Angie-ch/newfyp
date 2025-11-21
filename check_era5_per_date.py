"""
Check if ERA5 files match typhoon locations on specific dates
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr

print("="*80)
print("CHECKING IF ERA5 FILES MATCH TYPHOON LOCATIONS BY DATE")
print("="*80)
print()

# Load interpolated tracks
df = pd.read_csv('data/raw/interpolated_typhoon_tracks_2018_2021.csv', low_memory=False)

# Get a few storms
storm_ids = df['typhoon_id'].unique()[:5]

matches = 0
mismatches = 0

for storm_id in storm_ids:
    storm_df = df[df['typhoon_id'] == storm_id].copy()
    if len(storm_df) == 0:
        continue
    
    # Get storm track
    lats = storm_df['lat'].values
    lons = storm_df['lon'].values
    times = pd.to_datetime(storm_df['ISO_TIME'].values)
    
    # Get full track range (what the code currently requests)
    full_lat_range = (float(np.min(lats) - 10), float(np.max(lats) + 10))
    full_lon_range = (float(np.min(lons) - 10), float(np.max(lons) + 10))
    
    print(f"\nStorm: {storm_id}")
    print(f"  Full track range (what code requests):")
    print(f"    Lat: [{full_lat_range[0]:.2f}, {full_lat_range[1]:.2f}]")
    print(f"    Lon: [{full_lon_range[0]:.2f}, {full_lon_range[1]:.2f}]")
    
    # Check a few dates
    unique_dates = sorted(set(times.date))[:3]
    
    for date in unique_dates:
        date_str = date.strftime('%Y%m%d')
        year = date.year
        
        # Get typhoon location on this date
        date_mask = times.date == date
        if not date_mask.any():
            continue
        
        typhoon_lat = float(np.mean(lats[date_mask]))
        typhoon_lon = float(np.mean(lons[date_mask]))
        
        # Check if ERA5 file exists
        year_dir = Path(f"data/era5/ERA5_{year}_26data")
        era5_file = year_dir / f"era5_pl_{date_str}.nc"
        
        if not era5_file.exists():
            print(f"  {date_str}: File not found")
            continue
        
        # Get ERA5 file coverage
        try:
            ds = xr.open_dataset(era5_file)
            file_lon_min = float(ds.coords['longitude'].min().values)
            file_lon_max = float(ds.coords['longitude'].max().values)
            file_lat_min = float(ds.coords['latitude'].min().values)
            file_lat_max = float(ds.coords['latitude'].max().values)
            ds.close()
            
            # Check if typhoon location on this date is within file
            typhoon_in_file = (
                file_lon_min <= typhoon_lon <= file_lon_max and
                file_lat_min <= typhoon_lat <= file_lat_max
            )
            
            # Check if file overlaps with full track range (what code requests)
            file_overlaps_full_range = not (
                file_lon_max < full_lon_range[0] or file_lon_min > full_lon_range[1] or
                file_lat_max < full_lat_range[0] or file_lat_min > full_lat_range[1]
            )
            
            if typhoon_in_file:
                matches += 1
                status1 = "[MATCH]"
            else:
                mismatches += 1
                status1 = "[MISMATCH]"
            
            if file_overlaps_full_range:
                status2 = "[OVERLAPS full range]"
            else:
                status2 = "[OUTSIDE full range - THIS IS THE PROBLEM!]"
            
            print(f"  {date_str}: Typhoon at ({typhoon_lat:.2f}°N, {typhoon_lon:.2f}°E)")
            print(f"    File: lat=[{file_lat_min:.2f}, {file_lat_max:.2f}], lon=[{file_lon_min:.2f}, {file_lon_max:.2f}]")
            print(f"    {status1} - Typhoon location {'within' if typhoon_in_file else 'OUTSIDE'} file")
            print(f"    {status2}")
        
        except Exception as e:
            print(f"  {date_str}: ERROR - {e}")
            continue

print("\n" + "="*80)
print(f"SUMMARY: {matches} matches, {mismatches} mismatches")
print()
print("KEY INSIGHT:")
print("  If files match typhoon locations per date BUT don't overlap with")
print("  the full track range, then the code is requesting the WRONG spatial range!")
print("  Solution: Load files per date, not for entire track range.")










