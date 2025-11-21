"""Check spatial coverage of multiple ERA5 files"""
import xarray as xr
import numpy as np
from pathlib import Path

era5_2018_dir = Path("data/era5/ERA5_2018_26data")
era5_files = sorted(list(era5_2018_dir.glob("era5_pl_*.nc")))[:10]  # Check first 10 files

print("="*80)
print("ERA5 FILE SPATIAL COVERAGE CHECK")
print("="*80)

for era5_file in era5_files:
    try:
        ds = xr.open_dataset(era5_file, engine='h5netcdf')
        lat_coords = ds.coords['latitude'].values
        lon_coords = ds.coords['longitude'].values
        
        lat_min, lat_max = lat_coords.min(), lat_coords.max()
        lon_min, lon_max = lon_coords.min(), lon_coords.max()
        
        print(f"\n{era5_file.name}:")
        print(f"  Lat: {lat_min:.2f} to {lat_max:.2f} (range: {lat_max-lat_min:.2f}°)")
        print(f"  Lon: {lon_min:.2f} to {lon_max:.2f} (range: {lon_max-lon_min:.2f}°)")
        print(f"  Times: {len(ds.coords['valid_time'])}")
        
        ds.close()
    except Exception as e:
        print(f"\n{era5_file.name}: [ERROR] {e}")

print("\n" + "="*80)
print("REQUIRED COVERAGE FOR 2018 TYPHOONS:")
print("="*80)
print("  Lat: 4.40 to 45.84 (range: 41.44°)")
print("  Lon: 105.00 to 244.90 (range: 139.90°)")
print("\nConclusion: If all ERA5 files show similar small ranges (~10-20°),")
print("then the ERA5 data needs to be re-downloaded with full Pacific coverage.")
