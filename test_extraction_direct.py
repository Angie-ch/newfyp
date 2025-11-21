"""Test ERA5 extraction directly"""
from data.real_data_loader import ERA5Loader
import pandas as pd
import numpy as np

era5_loader = ERA5Loader()

# Load a test ERA5 dataset
start_time = pd.to_datetime('2018-08-01')
end_time = pd.to_datetime('2018-08-05')
era5_ds = era5_loader.load_era5_from_daily_files(
    start_time=start_time,
    end_time=end_time,
    lat_range=(0, 20),
    lon_range=(100, 140)
)

if era5_ds is None:
    print("ERROR: Failed to load ERA5")
    exit(1)

print(f"Dataset variables: {list(era5_ds.data_vars.keys())}")
print(f"Dataset dims: {dict(era5_ds.dims)}")

# Test extraction
test_lons = np.array([120.0, 121.0, 122.0])
test_lats = np.array([15.0, 15.5, 16.0])
test_times = pd.to_datetime(['2018-08-01 00:00', '2018-08-01 06:00', '2018-08-01 12:00'])

print(f"\nTesting extraction...")
print(f"Expected vars (pressure-level): {era5_loader.PRESSURE_LEVEL_VARS}")
print(f"Expected vars (single-level): {era5_loader.SINGLE_LEVEL_VARS}")
print(f"Reverse mapping: {era5_loader.REVERSE_VAR_MAPPING}")

# Test a single extraction step manually
import xarray as xr
time = test_times[0]
center_lon = test_lons[0]
center_lat = test_lats[0]
crop_size = 64
resolution = 0.25
box_size_deg = crop_size * resolution / 2

# Select time
ds_t = era5_ds.sel(valid_time=time, method='nearest')
print(f"\nAfter time selection, variables: {list(ds_t.data_vars.keys())}")

# Crop region
lat_slice = slice(center_lat + box_size_deg, center_lat - box_size_deg)
lon_slice = slice(center_lon - box_size_deg, center_lon + box_size_deg)
ds_crop = ds_t.sel(latitude=lat_slice, longitude=lon_slice)
print(f"After crop, variables: {list(ds_crop.data_vars.keys())}")
print(f"Crop shape check - z variable: {ds_crop['z'].shape if 'z' in ds_crop else 'NOT FOUND'}")

# Try extracting one variable
if 'z' in ds_crop:
    print(f"\nTesting 'z' variable extraction:")
    print(f"  dims: {ds_crop['z'].dims}")
    print(f"  shape: {ds_crop['z'].shape}")
    for level in era5_loader.PRESSURE_LEVELS:
        try:
            if 'pressure_level' in ds_crop['z'].dims:
                data = ds_crop['z'].sel(pressure_level=int(level)).values
                print(f"  Level {level}: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")
            else:
                print(f"  Level {level}: pressure_level dim not found")
        except Exception as e:
            print(f"  Level {level}: ERROR - {e}")

frames = era5_loader.extract_frames_at_times(
    era5_ds, test_lons, test_lats, test_times, crop_size=64
)

print(f"Extracted frames shape: {frames.shape}")
print(f"Frames range: {frames.min():.6f} to {frames.max():.6f}")
print(f"All zeros: {(frames == 0).all()}")
print(f"Non-zero count: {(frames != 0).sum()} / {frames.size}")

if (frames != 0).any():
    print("\n[OK] Extraction successful - contains real data")
else:
    print("\n[ERROR] Extraction failed - all zeros")

