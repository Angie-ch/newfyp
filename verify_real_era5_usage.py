"""
Verify if the code is using real ERA5 data from data/era5
"""
import sys
from pathlib import Path

# Add data directory to path
sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import ERA5Loader, IBTrACSLoader

def main():
    print("="*80)
    print("VERIFYING REAL ERA5 DATA USAGE")
    print("="*80)
    
    # Check 1: ERA5Loader path
    print("\n1. Checking ERA5Loader data directory...")
    era5_loader = ERA5Loader()
    print(f"   ERA5Loader.data_dir: {era5_loader.data_dir}")
    print(f"   Exists: {era5_loader.data_dir.exists()}")
    
    # Check 2: Look for ERA5 year directories
    print("\n2. Checking for ERA5 year directories...")
    era5_years = []
    for year in [2018, 2019, 2020, 2021]:
        year_dir = era5_loader.data_dir / f"ERA5_{year}_26data"
        exists = year_dir.exists()
        if exists:
            # Count .nc files
            nc_files = list(year_dir.glob("*.nc"))
            print(f"   ✓ ERA5_{year}_26data/ exists ({len(nc_files)} .nc files)")
            era5_years.append(year)
        else:
            print(f"   ✗ ERA5_{year}_26data/ NOT FOUND")
    
    if not era5_years:
        print("\n   ⚠️  WARNING: No ERA5 data directories found!")
        print(f"   Expected location: {era5_loader.data_dir}/ERA5_*_26data/")
        return False
    
    # Check 3: Test loading ERA5 data for a specific date
    print("\n3. Testing ERA5 data loading...")
    try:
        import pandas as pd
        from datetime import datetime
        
        # Try to load data for a specific date
        test_date = pd.Timestamp('2018-08-15 00:00:00')
        lat_range = (0, 20)
        lon_range = (100, 140)
        
        print(f"   Attempting to load ERA5 data for {test_date.date()}...")
        era5_ds = era5_loader.load_era5_from_daily_files(
            test_date, test_date, lat_range, lon_range
        )
        
        if era5_ds is not None:
            print(f"   ✓ Successfully loaded ERA5 dataset!")
            print(f"   Variables: {list(era5_ds.data_vars.keys())[:5]}...")
            print(f"   Dimensions: {dict(era5_ds.dims)}")
            return True
        else:
            print(f"   ✗ Failed to load ERA5 data for {test_date.date()}")
            print(f"   This might be normal if that specific date doesn't have data")
            return False
            
    except Exception as e:
        print(f"   ✗ Error loading ERA5 data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check 4: Check if processed data exists and verify it uses ERA5
    print("\n4. Checking processed dataset...")
    processed_dir = Path("data/processed_temporal_split")
    if processed_dir.exists():
        train_cases = processed_dir / "train" / "cases"
        if train_cases.exists():
            npz_files = list(train_cases.glob("*.npz"))
            if npz_files:
                print(f"   ✓ Found {len(npz_files)} processed .npz files")
                print(f"   These should contain real ERA5 data if generated correctly")
                return True
            else:
                print(f"   ⚠️  No .npz files found in processed dataset")
                return False
        else:
            print(f"   ⚠️  Processed cases directory not found")
            return False
    else:
        print(f"   ⚠️  Processed dataset directory not found")
        print(f"   Run data/generate_data_by_year.py to create processed dataset")
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "="*80)
    if success:
        print("✓ VERIFICATION COMPLETE: Real ERA5 data appears to be available")
    else:
        print("⚠️  VERIFICATION INCOMPLETE: Some issues detected")
    print("="*80)
    sys.exit(0 if success else 1)

