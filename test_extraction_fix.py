"""
Test the fixed ERA5 extraction with a real example
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import IBTrACSLoader, ERA5Loader

print("="*80)
print("TESTING FIXED ERA5 EXTRACTION")
print("="*80)
print()

# Load a storm
ibtracs_loader = IBTrACSLoader(data_dir="data/raw")
interpolated_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021.csv")

if not interpolated_file.exists():
    print(f"ERROR: {interpolated_file} not found")
    exit(1)

df = pd.read_csv(interpolated_file, low_memory=False)
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

# Get first storm
storm_ids = df['SID'].unique()[:1]
storm_id = storm_ids[0]

print(f"Testing with storm: {storm_id}")
storm_data = ibtracs_loader.get_storm_data(df, storm_id)
print(f"  Name: {storm_data['name']}")
print(f"  Track length: {len(storm_data['times'])} timesteps")
print()

# Load ERA5 data
era5_loader = ERA5Loader(data_dir="data/era5")

start_time = pd.to_datetime(storm_data['times'][0]) - pd.Timedelta(hours=6)
end_time = pd.to_datetime(storm_data['times'][-1]) + pd.Timedelta(hours=6)

lats = storm_data['lats']
lons = storm_data['lons']
lat_range = (float(np.min(lats) - 10), float(np.max(lats) + 10))
lon_range = (float(np.min(lons) - 10), float(np.max(lons) + 10))

print(f"Loading ERA5 data...")
print(f"  Time range: {start_time} to {end_time}")
print(f"  Spatial range: lat={lat_range}, lon={lon_range}")

era5_ds = era5_loader.load_era5_from_daily_files(
    start_time=start_time,
    end_time=end_time,
    lat_range=lat_range,
    lon_range=lon_range
)

if era5_ds is None:
    print("  [ERROR] Failed to load ERA5 dataset!")
    exit(1)

print(f"  [OK] ERA5 dataset loaded")
print(f"  Variables: {list(era5_ds.data_vars.keys())[:5]}...")
print(f"  Dimensions: {dict(era5_ds.dims)}")
print()

# Test extraction
print("Testing extraction...")
test_times = storm_data['times'][:5]  # First 5 timesteps
test_lons = storm_data['lons'][:5]
test_lats = storm_data['lats'][:5]

print(f"  Extracting {len(test_times)} timesteps...")

try:
    frames = era5_loader.extract_frames_at_times(
        era5_ds, test_lons, test_lats, test_times, crop_size=64
    )
    
    print()
    print("Extraction Results:")
    print(f"  Shape: {frames.shape}")
    print(f"  Range: [{frames.min():.4f}, {frames.max():.4f}]")
    print(f"  Mean: {frames.mean():.4f}")
    print(f"  Std: {frames.std():.4f}")
    print(f"  Non-zero: {(frames != 0).sum() / frames.size * 100:.2f}%")
    print(f"  Non-NaN: {(~np.isnan(frames)).sum() / frames.size * 100:.2f}%")
    print()
    
    if (frames == 0).all():
        print("  [ERROR] ALL ZEROS - Extraction failed!")
    elif np.isnan(frames).all():
        print("  [ERROR] ALL NaN - No data available!")
    else:
        print("  [OK] Extraction successful - contains real data!")
        
except Exception as e:
    print(f"  [ERROR] Extraction failed: {e}")
    import traceback
    traceback.print_exc()










