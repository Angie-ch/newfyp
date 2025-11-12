#!/usr/bin/env python3
"""Check ERA5 file structure"""
import xarray as xr

# Load a sample file
file_path = 'data/era5/ERA5_2021_26data/era5_pl_20210214.nc'
print(f"Loading: {file_path}\n")

ds = xr.open_dataset(file_path)

print("=== DATASET STRUCTURE ===")
print(f"Dimensions: {dict(ds.dims)}")
print(f"\nCoordinates: {list(ds.coords)}")
print(f"\nData variables: {list(ds.data_vars)}")

print("\n=== DETAILED VARIABLES ===")
for var in ds.data_vars:
    print(f"\n{var}:")
    print(f"  Shape: {ds[var].shape}")
    print(f"  Dims: {ds[var].dims}")
    print(f"  Dtype: {ds[var].dtype}")
    if hasattr(ds[var], 'long_name'):
        print(f"  Long name: {ds[var].long_name}")
    if hasattr(ds[var], 'units'):
        print(f"  Units: {ds[var].units}")

print("\n=== COORDINATE VALUES ===")
if 'level' in ds.coords:
    print(f"Pressure levels: {ds.level.values}")
if 'latitude' in ds.coords:
    print(f"Latitude range: [{ds.latitude.min().values:.2f}, {ds.latitude.max().values:.2f}]")
if 'longitude' in ds.coords:
    print(f"Longitude range: [{ds.longitude.min().values:.2f}, {ds.longitude.max().values:.2f}]")
if 'time' in ds.coords:
    print(f"Time steps: {len(ds.time)}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

ds.close()


