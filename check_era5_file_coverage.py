"""Check if ERA5 files are already cropped around typhoon centers or are regional files"""

import xarray as xr
import glob
from pathlib import Path
import numpy as np

# Check multiple files from different dates
era5_dir = Path("data/era5/ERA5_2018_26data")
files = sorted(glob.glob(str(era5_dir / "era5_pl_*.nc")))[:20]  # Check first 20 files

print("="*80)
print("CHECKING ERA5 FILE SPATIAL COVERAGE")
print("="*80)
print(f"\nChecking {len(files)} files to see if they're per-typhoon or regional...\n")

coverage_list = []
for f in files:
    try:
        ds = xr.open_dataset(f)
        lat_min = float(ds.latitude.min().values)
        lat_max = float(ds.latitude.max().values)
        lon_min = float(ds.longitude.min().values)
        lon_max = float(ds.longitude.max().values)
        lat_size = ds.dims.get('latitude', 0)
        lon_size = ds.dims.get('longitude', 0)
        
        coverage_list.append({
            'file': Path(f).name,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_size': lat_size,
            'lon_size': lon_size
        })
        
        print(f"{Path(f).name}:")
        print(f"  Lat: [{lat_min:.2f}°N, {lat_max:.2f}°N] ({lat_size} points)")
        print(f"  Lon: [{lon_min:.2f}°E, {lon_max:.2f}°E] ({lon_size} points)")
        print()
        
        ds.close()
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Analyze coverage
if coverage_list:
    lat_mins = [c['lat_min'] for c in coverage_list]
    lat_maxs = [c['lat_max'] for c in coverage_list]
    lon_mins = [c['lon_min'] for c in coverage_list]
    lon_maxs = [c['lon_max'] for c in coverage_list]
    
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print(f"\nLatitude range across all files:")
    print(f"  Min: {min(lat_mins):.2f}°N")
    print(f"  Max: {max(lat_maxs):.2f}°N")
    print(f"  Variation: {max(lat_maxs) - min(lat_mins):.2f}°")
    
    print(f"\nLongitude range across all files:")
    print(f"  Min: {min(lon_mins):.2f}°E")
    print(f"  Max: {max(lon_maxs):.2f}°E")
    print(f"  Variation: {max(lon_maxs) - min(lon_mins):.2f}°")
    
    # Check if all files have the same coverage (regional) or different (per-typhoon)
    lat_ranges = [(c['lat_min'], c['lat_max']) for c in coverage_list]
    lon_ranges = [(c['lon_min'], c['lon_max']) for c in coverage_list]
    
    unique_lat_ranges = len(set(lat_ranges))
    unique_lon_ranges = len(set(lon_ranges))
    
    print(f"\n{'='*80}")
    if unique_lat_ranges == 1 and unique_lon_ranges == 1:
        print("RESULT: REGIONAL FILES (all files have same coverage)")
        print("  → Files are NOT cropped around individual typhoons")
        print("  → They are pre-cropped to a fixed regional box")
        print("  → Regeneration is needed to crop around each typhoon center")
    else:
        print("RESULT: PER-TYPHOON FILES (files have different coverage)")
        print(f"  - Found {unique_lat_ranges} unique latitude ranges")
        print(f"  - Found {unique_lon_ranges} unique longitude ranges")
        print("  - Files appear to be already cropped around typhoons")
        print("  - May still need regeneration to combine into training samples")
    print("="*80)

