"""
Test if ERA5 extraction is working correctly now
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import ERA5Loader

print("="*80)
print("TESTING ERA5 EXTRACTION - IS IT WORKING?")
print("="*80)
print()

# Load ERA5 loader
era5_loader = ERA5Loader(data_dir="data/era5")

# Test with a known date and location that should have data
# Based on our earlier check, 2018-03-23 had a typhoon at (4.90°N, 146.69°E)
test_date = pd.Timestamp("2018-03-23")
test_lat = 4.90
test_lon = 146.69

print(f"Test date: {test_date.date()}")
print(f"Test location: ({test_lat}°N, {test_lon}°E)")
print()

# Load the ERA5 file for this date
year = test_date.year
date_str = test_date.strftime('%Y%m%d')
era5_file = Path(f"data/era5/ERA5_{year}_26data/era5_pl_{date_str}.nc")

if not era5_file.exists():
    print(f"[ERROR] File not found: {era5_file}")
    exit(1)

print(f"Loading file: {era5_file.name}")

# Load ERA5 data using the loader's method
start_time = test_date - pd.Timedelta(hours=6)
end_time = test_date + pd.Timedelta(hours=6)
lat_range = (test_lat - 10, test_lat + 10)
lon_range = (test_lon - 10, test_lon + 10)

era5_ds = era5_loader.load_era5_from_daily_files(
    start_time=start_time,
    end_time=end_time,
    lat_range=lat_range,
    lon_range=lon_range
)

if era5_ds is None:
    print("[ERROR] Failed to load ERA5 dataset!")
    exit(1)

print(f"[OK] Dataset loaded")
print(f"  Variables: {list(era5_ds.data_vars.keys())[:5]}...")
print(f"  Dimensions: {dict(era5_ds.dims)}")
print()

# Test extraction
print("Testing extraction...")
test_times = np.array([test_date])
test_lons = np.array([test_lon])
test_lats = np.array([test_lat])

try:
    frames = era5_loader.extract_frames_at_times(
        era5_ds, test_lons, test_lats, test_times, crop_size=64
    )
    
    print()
    print("="*80)
    print("EXTRACTION RESULTS:")
    print("="*80)
    print(f"  Shape: {frames.shape}")
    print(f"  Data type: {frames.dtype}")
    print(f"  Range: [{frames.min():.4f}, {frames.max():.4f}]")
    print(f"  Mean: {frames.mean():.4f}")
    print(f"  Std: {frames.std():.4f}")
    print(f"  Non-zero count: {(frames != 0).sum()} / {frames.size} ({(frames != 0).sum() / frames.size * 100:.2f}%)")
    print(f"  Non-NaN count: {(~np.isnan(frames)).sum()} / {frames.size} ({(~np.isnan(frames)).sum() / frames.size * 100:.2f}%)")
    print()
    
    # Check results
    all_zeros = (frames == 0).all()
    all_nan = np.isnan(frames).all()
    has_data = not all_zeros and not all_nan
    
    if all_zeros:
        print("  [FAIL] ALL ZEROS - Extraction returned zeros!")
        print("  Status: NOT WORKING")
    elif all_nan:
        print("  [FAIL] ALL NaN - No data available!")
        print("  Status: NOT WORKING")
    else:
        print("  [OK] Extraction successful!")
        print(f"  Contains real data: {has_data}")
        print(f"  Data range: [{frames[~np.isnan(frames)].min():.4f}, {frames[~np.isnan(frames)].max():.4f}]")
        print("  Status: WORKING!")
        
except Exception as e:
    print()
    print("="*80)
    print("[ERROR] Extraction failed!")
    print("="*80)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("Status: NOT WORKING")










