"""
Check if ERA5 files are cropped to match typhoon locations from IBTrACS
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
import sys

sys.path.append(str(Path(__file__).parent / "data"))
from real_data_loader import IBTrACSLoader

print("="*80)
print("CHECKING IF ERA5 FILES MATCH TYPHOON LOCATIONS")
print("="*80)
print()

# Load IBTrACS data
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

# Get a few storms
storm_ids = df['SID'].unique()[:5]

for storm_id in storm_ids:
    try:
        storm_data = ibtracs_loader.get_storm_data(df, storm_id)
        
        # Get storm track
        lats = storm_data['lats']
        lons = storm_data['lons']
        times = storm_data['times']
        
        # Get date range
        start_date = pd.to_datetime(times[0]).date()
        end_date = pd.to_datetime(times[-1]).date()
        
        print(f"\nStorm: {storm_data['name']} ({storm_id})")
        print(f"  Track: lat=[{np.min(lats):.2f}, {np.max(lats):.2f}], lon=[{np.min(lons):.2f}, {np.max(lons):.2f}]")
        print(f"  Time: {start_date} to {end_date}")
        
        # Check ERA5 files for this date range
        year = start_date.year
        year_dir = Path(f"data/era5/ERA5_{year}_26data")
        
        if not year_dir.exists():
            print(f"  [SKIP] No ERA5 directory for {year}")
            continue
        
        # Check a few dates along the track
        dates_to_check = pd.date_range(start=start_date, end=end_date, freq='2D')[:3]
        
        for date in dates_to_check:
            date_str = date.strftime('%Y%m%d')
            era5_file = year_dir / f"era5_pl_{date_str}.nc"
            
            if not era5_file.exists():
                continue
            
            # Get typhoon location on this date
            date_times = pd.to_datetime(times)
            idx = np.argmin(np.abs((date_times - date).total_seconds()))
            typhoon_lat = lats[idx]
            typhoon_lon = lons[idx]
            
            # Get ERA5 file coverage
            ds = xr.open_dataset(era5_file)
            file_lon_min = float(ds.coords['longitude'].min().values)
            file_lon_max = float(ds.coords['longitude'].max().values)
            file_lat_min = float(ds.coords['latitude'].min().values)
            file_lat_max = float(ds.coords['latitude'].max().values)
            ds.close()
            
            # Check if typhoon is within file coverage
            in_coverage = (
                file_lon_min <= typhoon_lon <= file_lon_max and
                file_lat_min <= typhoon_lat <= file_lat_max
            )
            
            status = "[OK]" if in_coverage else "[OUTSIDE]"
            print(f"  {date_str}: Typhoon at ({typhoon_lat:.2f}°N, {typhoon_lon:.2f}°E)")
            print(f"    File coverage: lat=[{file_lat_min:.2f}, {file_lat_max:.2f}], lon=[{file_lon_min:.2f}, {file_lon_max:.2f}]")
            print(f"    {status} - Typhoon {'within' if in_coverage else 'OUTSIDE'} file coverage")
    
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        continue










