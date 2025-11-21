"""
Check longitude range in ERA5 files
"""
import xarray as xr
from pathlib import Path

# Load a sample file
era5_file = Path("data/era5/ERA5_2018_26data/era5_pl_20180208.nc")
ds = xr.open_dataset(era5_file)

print("="*80)
print("CHECKING LONGITUDE RANGE IN ERA5 FILES")
print("="*80)
print()

print(f"File: {era5_file.name}")
print(f"Longitude coordinates:")
lon_coords = ds.coords['longitude'].values
print(f"  Min: {float(lon_coords.min())}")
print(f"  Max: {float(lon_coords.max())}")
print(f"  First 10: {lon_coords[:10]}")
print(f"  Last 10: {lon_coords[-10:]}")
print()

# Test selection with different longitude ranges
test_lon = 130.0  # Typical typhoon longitude
box_size = 8.0

print("Testing spatial selection:")
print(f"  Requested: lon = {test_lon} ± {box_size}")
print(f"  Range: [{test_lon - box_size}, {test_lon + box_size}]")
print()

# Try selection
lon_min = test_lon - box_size
lon_max = test_lon + box_size

try:
    ds_crop = ds.sel(longitude=slice(lon_min, lon_max))
    print(f"  Result: longitude dimension = {ds_crop.dims.get('longitude', 0)}")
    if ds_crop.dims.get('longitude', 0) == 0:
        print(f"  [ERROR] Empty selection - longitude range doesn't match!")
        print()
        print("  Possible issues:")
        print("    1. ERA5 uses 0-360 range, but code requests -180 to 180")
        print("    2. Longitude coordinates don't overlap with requested range")
        print("    3. Need to convert longitude or adjust selection")
    else:
        print(f"  [OK] Selection successful")
except Exception as e:
    print(f"  [ERROR] Selection failed: {e}")

ds.close()










