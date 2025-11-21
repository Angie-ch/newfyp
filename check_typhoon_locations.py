"""Check if typhoon locations match ERA5 spatial coverage"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load 6h resampled tracks
tracks_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021_6h.csv")
tracks_df = pd.read_csv(tracks_file)

# Filter for 2018
tracks_2018 = tracks_df[tracks_df['typhoon_id'].str.startswith('2018')]

# ERA5 file coverage (from the test file)
ERA5_LAT_RANGE = (9.9, -0.6)
ERA5_LON_RANGE = (141.4, 152.4)

print("="*80)
print("TYPHOON LOCATION vs ERA5 COVERAGE CHECK")
print("="*80)
print(f"\nERA5 file coverage:")
print(f"  Latitude: {ERA5_LAT_RANGE[0]} to {ERA5_LAT_RANGE[1]}")
print(f"  Longitude: {ERA5_LON_RANGE[0]} to {ERA5_LON_RANGE[1]}")

print(f"\n2018 Typhoon locations:")
print(f"  Total records: {len(tracks_2018)}")
print(f"  Unique typhoons: {tracks_2018['typhoon_id'].nunique()}")
print(f"\n  Latitude range: {tracks_2018['lat'].min():.2f} to {tracks_2018['lat'].max():.2f}")
print(f"  Longitude range: {tracks_2018['lon'].min():.2f} to {tracks_2018['lon'].max():.2f}")

# Check first few typhoons
print("\n" + "="*80)
print("SAMPLE TYPHOON LOCATIONS:")
print("="*80)
for typhoon_id in tracks_2018['typhoon_id'].unique()[:5]:
    storm_data = tracks_2018[tracks_2018['typhoon_id'] == typhoon_id]
    lat_min, lat_max = storm_data['lat'].min(), storm_data['lat'].max()
    lon_min, lon_max = storm_data['lon'].min(), storm_data['lon'].max()
    
    # Check if within ERA5 bounds
    lat_in_bounds = ERA5_LAT_RANGE[1] <= lat_min and lat_max <= ERA5_LAT_RANGE[0]
    lon_in_bounds = ERA5_LON_RANGE[0] <= lon_min and lon_max <= ERA5_LON_RANGE[1]
    
    status = "[OK]" if (lat_in_bounds and lon_in_bounds) else "[OUT OF BOUNDS]"
    
    print(f"\n{typhoon_id}: {status}")
    print(f"  Lat: {lat_min:.2f} to {lat_max:.2f} (ERA5: {ERA5_LAT_RANGE[1]} to {ERA5_LAT_RANGE[0]})")
    print(f"  Lon: {lon_min:.2f} to {lon_max:.2f} (ERA5: {ERA5_LON_RANGE[0]} to {ERA5_LON_RANGE[1]})")
    print(f"  Records: {len(storm_data)}")

