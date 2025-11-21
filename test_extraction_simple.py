"""
Simple test of ERA5 extraction to find the issue
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import ERA5Loader

# Load a sample ERA5 file
era5_file = Path("data/era5/ERA5_2018_26data/era5_pl_20180208.nc")
print(f"Loading: {era5_file.name}")
ds = xr.open_dataset(era5_file)

print(f"\nDataset structure:")
print(f"  Variables: {list(ds.data_vars.keys())}")
print(f"  Dimensions: {dict(ds.dims)}")
print(f"  Coordinates: {list(ds.coords.keys())}")

# Check a variable
var_name = 'z'
if var_name in ds.data_vars:
    var = ds[var_name]
    print(f"\nVariable '{var_name}':")
    print(f"  dims: {var.dims}")
    print(f"  shape: {var.shape}")
    print(f"  coords: {list(var.coords.keys())}")
    
    # Check pressure level dimension
    if 'pressure_level' in var.coords:
        levels = var.coords['pressure_level'].values
        print(f"  pressure_level values: {levels}")
    elif 'level' in var.coords:
        levels = var.coords['level'].values
        print(f"  level values: {levels}")
    else:
        print(f"  No level dimension found!")
        print(f"  All dims: {var.dims}")

# Test spatial selection
print(f"\nTesting spatial selection...")
test_lat = 20.0
test_lon = 130.0
box_size = 8.0  # degrees

lat_min = test_lat - box_size
lat_max = test_lat + box_size
lon_min = test_lon - box_size
lon_max = test_lon + box_size

try:
    # Check latitude order
    lat_coords = ds.coords['latitude'].values
    if len(lat_coords) > 1 and lat_coords[0] > lat_coords[-1]:
        lat_slice = slice(lat_max, lat_min)  # Descending
    else:
        lat_slice = slice(lat_min, lat_max)  # Ascending
    
    ds_crop = ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    print(f"  Crop successful!")
    print(f"  Cropped shape: {dict(ds_crop.dims)}")
    
    # Try extracting z variable
    if 'z' in ds_crop:
        z_var = ds_crop['z']
        print(f"\n  Variable z in crop:")
        print(f"    dims: {z_var.dims}")
        print(f"    shape: {z_var.shape}")
        
        # Try selecting a pressure level
        if 'pressure_level' in z_var.dims:
            try:
                level_data = z_var.isel(pressure_level=0)  # First level
                if 'valid_time' in level_data.dims:
                    level_data = level_data.isel(valid_time=0)
                print(f"    Selected level 0: shape={level_data.shape}, range=[{float(level_data.min().values):.2f}, {float(level_data.max().values):.2f}]")
                print(f"    [OK] Extraction works!")
            except Exception as e:
                print(f"    ERROR selecting level: {e}")
        else:
            print(f"    No pressure_level dimension!")
            print(f"    Available dims: {z_var.dims}")
    
except Exception as e:
    print(f"  ERROR in spatial selection: {e}")
    import traceback
    traceback.print_exc()

ds.close()










