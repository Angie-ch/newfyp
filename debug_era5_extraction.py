"""Debug why ERA5 extraction returns all NaN"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# Load one ERA5 file directly
era5_file = Path("data/era5/ERA5_2018_26data/era5_pl_20180324.nc")
ds = xr.open_dataset(era5_file, engine='h5netcdf')

print("="*80)
print("DEBUGGING ERA5 EXTRACTION")
print("="*80)

print("\n1. RAW DATASET INFO:")
print("-"*80)
print(ds)

print("\n2. DATASET DIMENSIONS:")
print("-"*80)
for dim in ds.dims:
    print(f"  {dim}: {ds.dims[dim]}")

print("\n3. COORDINATES:")
print("-"*80)
for coord in ds.coords:
    vals = ds.coords[coord].values
    try:
        if hasattr(vals, '__len__') and len(vals) > 5:
            print(f"  {coord}: [{vals[0]}, {vals[1]}, ..., {vals[-2]}, {vals[-1]}] (len={len(vals)})")
        else:
            print(f"  {coord}: {vals}")
    except:
        print(f"  {coord}: {vals} (scalar)")

print("\n4. DATA VARIABLES:")
print("-"*80)
for var in ds.data_vars:
    data = ds[var].values
    print(f"\n  {var}:")
    print(f"    Shape: {data.shape}")
    print(f"    Dtype: {data.dtype}")
    print(f"    Min: {np.nanmin(data):.2f}, Max: {np.nanmax(data):.2f}")
    print(f"    NaN count: {np.isnan(data).sum()} / {data.size}")
    print(f"    Sample (first timestep, first level, center pixel):")
    if len(data.shape) == 4:  # (time, level, lat, lon)
        center_lat_idx = data.shape[2] // 2
        center_lon_idx = data.shape[3] // 2
        print(f"      {data[0, 0, center_lat_idx, center_lon_idx]}")

print("\n5. TESTING SPATIAL EXTRACTION:")
print("-"*80)

# Try to extract data at a specific location
center_lat = 5.78  # From typhoon position on March 24
center_lon = 143.86

print(f"Target location: ({center_lat}, {center_lon})")

# Find nearest indices
lat_coord = ds.coords['latitude'].values
lon_coord = ds.coords['longitude'].values

print(f"  Latitude range in file: {lat_coord.min():.2f} to {lat_coord.max():.2f}")
print(f"  Longitude range in file: {lon_coord.min():.2f} to {lon_coord.max():.2f}")

# Check if target is in range
if lat_coord.min() <= center_lat <= lat_coord.max() and \
   lon_coord.min() <= center_lon <= lon_coord.max():
    print(f"  [OK] Target location IS within ERA5 bounds")
    
    # Find nearest indices
    lat_idx = np.argmin(np.abs(lat_coord - center_lat))
    lon_idx = np.argmin(np.abs(lon_coord - center_lon))
    
    print(f"  Nearest lat: {lat_coord[lat_idx]:.2f} (index {lat_idx})")
    print(f"  Nearest lon: {lon_coord[lon_idx]:.2f} (index {lon_idx})")
    
    # Extract data at this location
    print(f"\n  Data at this location:")
    for var in list(ds.data_vars)[:3]:  # Check first 3 variables
        data_at_loc = ds[var].values[0, 0, lat_idx, lon_idx]  # First time, first level
        print(f"    {var}: {data_at_loc}")
else:
    print(f"  [ERROR] Target location is OUT OF BOUNDS!")

print("\n6. TESTING TIME EXTRACTION:")
print("-"*80)

target_time = pd.Timestamp('2018-03-24 12:00:00')
print(f"Target time: {target_time}")

time_coord = pd.to_datetime(ds.coords['valid_time'].values)
print(f"  Available times: {time_coord}")

# Check if target time exists
if target_time in time_coord:
    print(f"  [OK] Target time EXISTS in ERA5 data")
    time_idx = list(time_coord).index(target_time)
    print(f"  Time index: {time_idx}")
    
    # Extract data at this time
    print(f"\n  Data at center location and target time:")
    for var in list(ds.data_vars)[:3]:
        data_at_time = ds[var].values[time_idx, 0, lat_idx, lon_idx]
        print(f"    {var}: {data_at_time}")
else:
    print(f"  [WARNING] Target time NOT found in ERA5 data")
    print(f"  Closest time: {time_coord[np.argmin(np.abs(time_coord - target_time))]}")

ds.close()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\nIf data extraction shows valid values here, the problem is in the")
print("extract_frames_at_times() function logic, not the ERA5 files themselves.")
