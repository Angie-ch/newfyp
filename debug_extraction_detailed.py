"""
Debug ERA5 extraction in detail to find where it fails
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import ERA5Loader

print("="*80)
print("DEBUGGING ERA5 EXTRACTION IN DETAIL")
print("="*80)
print()

# Load ERA5 loader
era5_loader = ERA5Loader(data_dir="data/era5")

# Test with a known date
test_date = pd.Timestamp("2018-03-23")
test_lat = 4.90
test_lon = 146.69

# Load the file directly
year = test_date.year
date_str = test_date.strftime('%Y%m%d')
era5_file = Path(f"data/era5/ERA5_{year}_26data/era5_pl_{date_str}.nc")

print(f"Loading file: {era5_file.name}")
ds = xr.open_dataset(era5_file)

print(f"\nFile structure:")
print(f"  Variables: {list(ds.data_vars.keys())}")
print(f"  Dimensions: {dict(ds.dims)}")
print(f"  Coordinates: {list(ds.coords.keys())}")
print()

# Check spatial coverage
lon_coords = ds.coords['longitude'].values
lat_coords = ds.coords['latitude'].values
print(f"Spatial coverage:")
print(f"  Longitude: [{lon_coords.min():.2f}, {lon_coords.max():.2f}]")
print(f"  Latitude: [{lat_coords.min():.2f}, {lat_coords.max():.2f}]")
print(f"  Typhoon location: ({test_lat:.2f}°N, {test_lon:.2f}°E)")
print(f"  In coverage: {lon_coords.min() <= test_lon <= lon_coords.max() and lat_coords.min() <= test_lat <= lat_coords.max()}")
print()

# Check time dimension
time_dim = None
for dim_name in ['time', 'valid_time', 't']:
    if dim_name in ds.dims or dim_name in ds.coords:
        time_dim = dim_name
        break

print(f"Time dimension: {time_dim}")
if time_dim:
    time_coords = ds.coords[time_dim].values
    print(f"  Time values: {time_coords}")
    print(f"  Requested time: {test_date}")
    # Find nearest time
    if hasattr(time_coords, 'values'):
        time_values = pd.to_datetime(time_coords)
        time_idx = np.argmin(np.abs((time_values - test_date).total_seconds()))
        nearest_time = time_values[time_idx]
        print(f"  Nearest time: {nearest_time} (index {time_idx})")
print()

# Try selecting time and location
print("Testing spatial selection...")
box_size = 8.0  # degrees (64 pixels * 0.25 resolution / 2)
lat_min = test_lat - box_size
lat_max = test_lat + box_size
lon_min = test_lon - box_size
lon_max = test_lon + box_size

print(f"  Crop region: lat=[{lat_min:.2f}, {lat_max:.2f}], lon=[{lon_min:.2f}, {lon_max:.2f}]")

# Select time
if time_dim:
    try:
        ds_t = ds.sel({time_dim: test_date}, method='nearest')
        print(f"  [OK] Time selection successful")
    except Exception as e:
        print(f"  [ERROR] Time selection failed: {e}")
        ds_t = ds.isel({time_dim: 0})  # Use first time
        print(f"  Using first time slice instead")
else:
    ds_t = ds

# Select spatial region
try:
    # Check latitude order
    if len(lat_coords) > 1 and lat_coords[0] > lat_coords[-1]:
        lat_slice = slice(lat_max, lat_min)  # Descending
    else:
        lat_slice = slice(lat_min, lat_max)  # Ascending
    
    ds_crop = ds_t.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max), method='nearest')
    print(f"  [OK] Spatial selection successful")
    print(f"  Cropped dimensions: {dict(ds_crop.dims)}")
    
    # Check if we have data
    if len(ds_crop.data_vars) > 0:
        first_var = list(ds_crop.data_vars.keys())[0]
        data_sample = ds_crop[first_var].values
        print(f"  Sample data from '{first_var}':")
        print(f"    Shape: {data_sample.shape}")
        print(f"    Range: [{np.nanmin(data_sample):.4f}, {np.nanmax(data_sample):.4f}]")
        print(f"    Non-zero: {(data_sample != 0).sum()} / {data_sample.size}")
        print(f"    Non-NaN: {(~np.isnan(data_sample)).sum()} / {data_sample.size}")
        
        if (data_sample == 0).all():
            print(f"  [WARNING] All zeros in sample!")
        elif np.isnan(data_sample).all():
            print(f"  [WARNING] All NaN in sample!")
        else:
            print(f"  [OK] Sample contains real data!")
    
except Exception as e:
    print(f"  [ERROR] Spatial selection failed: {e}")
    import traceback
    traceback.print_exc()

ds.close()










