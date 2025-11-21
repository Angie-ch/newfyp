"""Test data generation for the single typhoon with ERA5 data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.real_data_loader import IBTrACSLoader, ERA5Loader
import pandas as pd
import numpy as np

# Configuration
TYPHOON_ID = '2018082N04147'
DATA_DIR = Path("data")

print("="*80)
print(f"TESTING SINGLE TYPHOON: {TYPHOON_ID}")
print("="*80)

# Load IBTrACS data
ibtracs_loader = IBTrACSLoader(data_dir=DATA_DIR / "raw")
interpolated_file = DATA_DIR / "raw" / "interpolated_typhoon_tracks_2018_2021_6h.csv"
df = pd.read_csv(interpolated_file)

# The 6h file uses 'typhoon_id' instead of 'SID', so get the data manually
storm_df = df[df['typhoon_id'] == TYPHOON_ID].copy()
storm_df = storm_df.sort_values('ISO_TIME')
storm_data = {
    'storm_id': TYPHOON_ID,
    'times': storm_df['ISO_TIME'].tolist(),
    'lats': storm_df['lat'].tolist(),
    'lons': storm_df['lon'].tolist(),
    'max_sustained_wind': storm_df['USA_WIND'].tolist() if 'USA_WIND' in storm_df.columns else [np.nan]*len(storm_df),
    'min_pressure': storm_df['USA_PRES'].tolist() if 'USA_PRES' in storm_df.columns else [np.nan]*len(storm_df)
}
print(f"\nStorm info:")
print(f"  Times: {len(storm_data['times'])} timesteps")
print(f"  Start: {pd.to_datetime(storm_data['times'][0])}")
print(f"  End: {pd.to_datetime(storm_data['times'][-1])}")
print(f"  Lat range: {min(storm_data['lats']):.2f} to {max(storm_data['lats']):.2f}")
print(f"  Lon range: {min(storm_data['lons']):.2f} to {max(storm_data['lons']):.2f}")

# Try loading ERA5 for different timesteps
print("\n" + "="*80)
print("TESTING ERA5 LOADING FOR DIFFERENT TIMESTEPS:")
print("="*80)

era5_loader = ERA5Loader(data_dir=DATA_DIR / "era5")

# Test timestep 0 (earliest - likely to fail due to needing past data)
print("\n[TEST 1] Timestep 0 (earliest, needs past data):")
print("-"*80)
test_idx = 0
try:
    center_time = pd.to_datetime(storm_data['times'][test_idx])
    center_lat = storm_data['lats'][test_idx]
    center_lon = storm_data['lons'][test_idx]
    
    print(f"Center time: {center_time}")
    print(f"Center position: ({center_lat:.2f}, {center_lon:.2f})")
    
    # Need 4 past timesteps (24 hours) + current + 8 future (48 hours)
    start_time = center_time - pd.Timedelta(hours=24)
    end_time = center_time + pd.Timedelta(hours=48)
    
    print(f"Required ERA5 time range: {start_time} to {end_time}")
    
    lat_range = (center_lat - 10, center_lat + 10)
    lon_range = (center_lon - 10, center_lon + 10)
    
    ds = era5_loader.load_era5_from_daily_files(
        start_time=start_time,
        end_time=end_time,
        lat_range=lat_range,
        lon_range=lon_range
    )
    
    if ds is not None:
        print(f"[OK] ERA5 loaded successfully!")
        print(f"  Dataset shape: {ds.dims}")
        ds.close()
    else:
        print(f"[FAILED] ERA5 returned None")
except Exception as e:
    print(f"[ERROR] {e}")

# Test timestep 5 (later, after we have 24h of history)
print("\n[TEST 2] Timestep 5 (should have enough history):")
print("-"*80)
test_idx = 5
try:
    center_time = pd.to_datetime(storm_data['times'][test_idx])
    center_lat = storm_data['lats'][test_idx]
    center_lon = storm_data['lons'][test_idx]
    
    print(f"Center time: {center_time}")
    print(f"Center position: ({center_lat:.2f}, {center_lon:.2f})")
    
    start_time = center_time - pd.Timedelta(hours=24)
    end_time = center_time + pd.Timedelta(hours=48)
    
    print(f"Required ERA5 time range: {start_time} to {end_time}")
    
    lat_range = (center_lat - 10, center_lat + 10)
    lon_range = (center_lon - 10, center_lon + 10)
    
    ds = era5_loader.load_era5_from_daily_files(
        start_time=start_time,
        end_time=end_time,
        lat_range=lat_range,
        lon_range=lon_range
    )
    
    if ds is not None:
        print(f"[OK] ERA5 loaded successfully!")
        print(f"  Dataset dims: {ds.dims}")
        print(f"  Time coords: {len(ds.coords.get('valid_time', ds.coords.get('time', [])))}")
        
        # Try extracting frames
        times_to_extract = [
            center_time - pd.Timedelta(hours=24),
            center_time - pd.Timedelta(hours=18),
            center_time - pd.Timedelta(hours=12),
            center_time - pd.Timedelta(hours=6),
            center_time
        ]
        
        print(f"\n  Trying to extract frames at {len(times_to_extract)} timesteps...")
        
        frames = era5_loader.extract_frames_at_times(
            ds=ds,
            times=times_to_extract,
            center_lats=[center_lat] * len(times_to_extract),
            center_lons=[center_lon] * len(times_to_extract)
        )
        
        if frames is not None:
            print(f"  [OK] Extracted frames: shape {frames.shape}")
            print(f"  [OK] Data range: [{np.nanmin(frames):.2f}, {np.nanmax(frames):.2f}]")
            print(f"  [OK] NaN count: {np.isnan(frames).sum()} / {frames.size}")
        else:
            print(f"  [FAILED] Frame extraction returned None")
        
        ds.close()
    else:
        print(f"[FAILED] ERA5 returned None")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

