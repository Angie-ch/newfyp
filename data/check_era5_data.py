#!/usr/bin/env python3
"""
Standalone script to check if all required ERA5 data files exist

Usage:
    python data/check_era5_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from generate_data_by_year import check_all_era5_data_exists
from real_data_loader import IBTrACSLoader

if __name__ == "__main__":
    print("="*80)
    print("ERA5 DATA AVAILABILITY CHECK")
    print("="*80)
    
    # Configuration
    START_YEAR = 2018
    END_YEAR = 2021
    
    # Initialize loader
    SCRIPT_DIR = Path(__file__).parent
    ibtracs_loader = IBTrACSLoader(
        data_dir=str(SCRIPT_DIR / "raw")
    )
    
    # Load IBTrACS data
    interpolated_file = SCRIPT_DIR / "raw" / "interpolated_typhoon_tracks_2018_2021.csv"
    if interpolated_file.exists():
        print(f"Loading interpolated tracks from {interpolated_file}")
        import pandas as pd
        df = pd.read_csv(interpolated_file, low_memory=False)
        print(f"✓ Loaded {len(df)} interpolated records")
        
        # Map column names
        if 'typhoon_id' in df.columns:
            df['SID'] = df['typhoon_id']
        if 'typhoon_name' in df.columns:
            df['NAME'] = df['typhoon_name']
        if 'lat' in df.columns:
            df['LAT'] = df['lat']
        if 'lon' in df.columns:
            df['LON'] = df['lon']
        if 'wind' in df.columns:
            df['WMO_WIND'] = df['wind']
            df['USA_WIND'] = df['wind'] / 0.514444
        if 'pressure' in df.columns:
            df['WMO_PRES'] = df['pressure']
            df['USA_PRES'] = df['pressure']
    else:
        print(f"Interpolated file not found, using standard IBTrACS data")
        df = ibtracs_loader.load_ibtracs()
    
    # Filter typhoons
    storm_ids = ibtracs_loader.filter_typhoons(
        df,
        start_year=START_YEAR,
        end_year=END_YEAR,
        min_wind_speed=33.0,
        min_duration_hours=48
    )
    
    print(f"\nFound {len(storm_ids)} storms to check")
    
    # Run the check
    results = check_all_era5_data_exists(
        ibtracs_loader=ibtracs_loader,
        df=df,
        storm_ids=storm_ids,
        verbose=True
    )
    
    # Exit with appropriate code
    if results['all_exist']:
        print("\n✓ All ERA5 data is available. Ready to generate dataset.")
        sys.exit(0)
    else:
        print("\n✗ Some ERA5 data is missing. Please download missing files before generating dataset.")
        sys.exit(1)

