"""
Debug why ERA5 extraction is returning zeros
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import IBTrACSLoader, ERA5Loader

def main():
    print("="*80)
    print("DEBUGGING ERA5 EXTRACTION ISSUE")
    print("="*80)
    print()
    
    # Load a storm
    print("1. Loading IBTrACS data...")
    ibtracs_loader = IBTrACSLoader(data_dir="data/raw")
    interpolated_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021.csv")
    
    if not interpolated_file.exists():
        print(f"ERROR: {interpolated_file} not found")
        return
    
    df = pd.read_csv(interpolated_file, low_memory=False)
    if 'typhoon_id' in df.columns:
        df['SID'] = df['typhoon_id']
    if 'typhoon_name' in df.columns:
        df['NAME'] = df['typhoon_name']
    if 'lat' in df.columns:
        df['LAT'] = df['lat']
    if 'lon' in df.columns:
        df['LON'] = df['lon']
    
    # Get first storm
    storm_ids = df['SID'].unique()[:1]
    storm_id = storm_ids[0]
    print(f"   Using storm: {storm_id}")
    
    storm_data = ibtracs_loader.get_storm_data(df, storm_id)
    print(f"   Storm: {storm_data['name']}")
    print(f"   Times: {len(storm_data['times'])} timesteps")
    print()
    
    # Load ERA5 data
    print("2. Loading ERA5 data...")
    era5_loader = ERA5Loader(data_dir="data/era5")
    
    start_time = pd.to_datetime(storm_data['times'][0]) - pd.Timedelta(hours=6)
    end_time = pd.to_datetime(storm_data['times'][-1]) + pd.Timedelta(hours=6)
    
    lats = storm_data['lats']
    lons = storm_data['lons']
    lat_range = (float(np.min(lats) - 10), float(np.max(lats) + 10))
    lon_range = (float(np.min(lons) - 10), float(np.max(lons) + 10))
    
    print(f"   Time range: {start_time} to {end_time}")
    print(f"   Spatial range: lat={lat_range}, lon={lon_range}")
    
    era5_ds = era5_loader.load_era5_from_daily_files(
        start_time=start_time,
        end_time=end_time,
        lat_range=lat_range,
        lon_range=lon_range
    )
    
    if era5_ds is None:
        print("   ERROR: Failed to load ERA5 dataset!")
        return
    
    print(f"   ERA5 dataset loaded")
    print(f"   Variables: {list(era5_ds.data_vars.keys())}")
    print(f"   Dimensions: {dict(era5_ds.dims)}")
    print()
    
    # Try extraction
    print("3. Testing extraction...")
    test_times = storm_data['times'][:3]
    test_lons = storm_data['lons'][:3]
    test_lats = storm_data['lats'][:3]
    
    print(f"   Extracting {len(test_times)} timesteps...")
    print(f"   Locations: {list(zip(test_lats, test_lons))}")
    
    frames = era5_loader.extract_frames_at_times(
        era5_ds, test_lons, test_lats, test_times, crop_size=64
    )
    
    print()
    print("4. Extraction Results:")
    print(f"   Shape: {frames.shape}")
    print(f"   Range: [{frames.min():.4f}, {frames.max():.4f}]")
    print(f"   Mean: {frames.mean():.4f}")
    print(f"   Std: {frames.std():.4f}")
    print(f"   Non-zero: {(frames != 0).sum() / frames.size * 100:.2f}%")
    print()
    
    if (frames == 0).all():
        print("   [ERROR] ALL ZEROS - Extraction failed!")
        print()
        print("   Debugging why...")
        print("   Checking dataset structure...")
        print(f"   Time dimension: {[d for d in era5_ds.dims if 'time' in d.lower()]}")
        print(f"   Coordinate names: {list(era5_ds.coords.keys())}")
        print(f"   Data variable names: {list(era5_ds.data_vars.keys())}")
        
        # Check if variables match expected names
        print()
        print("   Expected variables:")
        print(f"     Single-level: {era5_loader.SINGLE_LEVEL_VARS}")
        print(f"     Pressure-level: {era5_loader.PRESSURE_LEVEL_VARS}")
        print()
        print("   Actual variables in dataset:")
        for var in list(era5_ds.data_vars.keys())[:10]:
            print(f"     {var}: {era5_ds[var].dims}, shape={era5_ds[var].shape}")
    else:
        print("   [OK] Extraction successful - contains real data!")

if __name__ == "__main__":
    main()










