"""Test script to inspect ERA5 file contents"""
import xarray as xr
import numpy as np
from pathlib import Path

# Load one ERA5 file to inspect
era5_file = Path("data/era5/ERA5_2018_26data/era5_pl_20180323.nc")

if era5_file.exists():
    print(f"Opening: {era5_file}")
    print(f"File size: {era5_file.stat().st_size} bytes\n")
    
    try:
        ds = xr.open_dataset(era5_file, engine='h5netcdf')
        print("Dataset loaded successfully!\n")
        print("="*80)
        print("DATASET INFO:")
        print("="*80)
        print(ds)
        print("\n" + "="*80)
        print("COORDINATES:")
        print("="*80)
        for coord in ds.coords:
            vals = ds.coords[coord].values
            if hasattr(vals, '__len__') and len(vals) > 1:
                print(f"{coord}: {vals[:5]}... (shape: {ds.coords[coord].shape})")
            else:
                print(f"{coord}: {vals} (scalar)")
        
        print("\n" + "="*80)
        print("DATA VARIABLES:")
        print("="*80)
        for var in ds.data_vars:
            data = ds[var].values
            print(f"\n{var}:")
            print(f"  Shape: {data.shape}")
            print(f"  Min: {np.nanmin(data):.2f}, Max: {np.nanmax(data):.2f}")
            print(f"  Mean: {np.nanmean(data):.2f}")
            print(f"  NaN count: {np.isnan(data).sum()}")
            print(f"  Zero count: {(data == 0).sum()}")
            print(f"  Sample values: {data.flat[:5]}")
        
        ds.close()
        print("\n[OK] ERA5 file is valid and contains data!")
        
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
else:
    print(f"[ERROR] File not found: {era5_file}")

