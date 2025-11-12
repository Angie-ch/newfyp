#!/usr/bin/env python3
"""
Simple test script to verify ERA5 data is properly formatted.
Tests only the pressure-level (pl) files which are used by the pipeline.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def test_era5_file(file_path):
    """Test a single ERA5 file."""
    try:
        ds = xr.open_dataset(file_path)
        
        # Check dimensions
        expected_dims = ['valid_time', 'pressure_level', 'latitude', 'longitude']
        missing_dims = [d for d in expected_dims if d not in ds.dims]
        if missing_dims:
            return False, f"Missing dimensions: {missing_dims}"
        
        # Check for NaN/Inf
        issues = []
        for var_name in ds.data_vars:
            data = ds[var_name].values
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            if nan_count > 0 or inf_count > 0:
                issues.append(f"{var_name}: {nan_count} NaN, {inf_count} Inf")
        
        # Get basic info
        info = {
            'timesteps': len(ds['valid_time']),
            'pressure_levels': ds['pressure_level'].values.tolist(),
            'lat_range': [float(ds['latitude'].values.min()), float(ds['latitude'].values.max())],
            'lon_range': [float(ds['longitude'].values.min()), float(ds['longitude'].values.max())],
            'variables': list(ds.data_vars.keys()),
            'issues': issues
        }
        
        ds.close()
        return True, info
        
    except Exception as e:
        return False, str(e)


def main():
    era5_dir = Path('data/era5')
    
    if not era5_dir.exists():
        print(f"✗ ERA5 directory not found: {era5_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("ERA5 Pressure-Level Data Verification")
    print("=" * 70)
    
    # Find all year directories
    year_dirs = sorted([d for d in era5_dir.iterdir() if d.is_dir() and 'ERA5' in d.name])
    
    if not year_dirs:
        print(f"✗ No ERA5 data directories found in {era5_dir}")
        sys.exit(1)
    
    all_valid = True
    total_files = 0
    
    for year_dir in year_dirs:
        # Only test pressure-level files (era5_pl_*.nc)
        pl_files = sorted(year_dir.glob("era5_pl_*.nc"))
        
        if not pl_files:
            print(f"\n{year_dir.name}:")
            print("  ⚠ No pressure-level files found (era5_pl_*.nc)")
            continue
        
        print(f"\n{year_dir.name}:")
        print(f"  Files: {len(pl_files)} pressure-level files")
        total_files += len(pl_files)
        
        # Test a few files
        test_indices = [0, len(pl_files)//2, -1]
        if len(pl_files) < 3:
            test_indices = [0]
        
        for idx in test_indices:
            file_path = pl_files[idx]
            valid, result = test_era5_file(file_path)
            
            if valid:
                info = result
                print(f"  ✓ {file_path.name}")
                print(f"      Timesteps: {info['timesteps']}, "
                      f"Pressure levels: {info['pressure_levels']}")
                print(f"      Lat: [{info['lat_range'][0]:.1f}, {info['lat_range'][1]:.1f}], "
                      f"Lon: [{info['lon_range'][0]:.1f}, {info['lon_range'][1]:.1f}]")
                print(f"      Variables: {', '.join(info['variables'])}")
                if info['issues']:
                    print(f"      ⚠ Issues: {'; '.join(info['issues'])}")
                    all_valid = False
            else:
                print(f"  ✗ {file_path.name}: {result}")
                all_valid = False
    
    print("\n" + "=" * 70)
    print(f"Summary: {total_files} total pressure-level files")
    
    if all_valid:
        print("\n✓✓✓ All ERA5 pressure-level files are valid!")
        print("\nYour ERA5 data is ready to use with the pipeline.")
        print("\nTo generate samples with 8 past + 12 future timesteps:")
        print("  python run_real_data_pipeline.py --n-samples 100")
        print("\nConfiguration:")
        print("  - Input: 8 timesteps (48 hours at 6h intervals)")
        print("  - Output: 12 timesteps (72 hours at 6h intervals)")
        print("  - Total: 120 hours (5 days) per sample")
        return 0
    else:
        print("\n⚠ Some files have issues (see above)")
        return 1


if __name__ == '__main__':
    sys.exit(main())

