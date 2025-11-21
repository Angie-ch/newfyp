"""
Diagnose why ERA5 data might not be loading correctly
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from data.real_data_loader import IBTrACSLoader, ERA5Loader
import pandas as pd

def main():
    print("="*80)
    print("DIAGNOSING ERA5 DATA LOADING")
    print("="*80)
    print()
    
    # Check 1: ERA5 directories exist
    print("1. Checking ERA5 directories...")
    era5_base = Path("data/era5")
    if not era5_base.exists():
        print(f"   [ERROR] ERA5 base directory not found: {era5_base}")
        return
    
    print(f"   [OK] ERA5 base directory exists: {era5_base}")
    
    for year in [2018, 2019, 2020, 2021]:
        year_dir = era5_base / f"ERA5_{year}_26data"
        if year_dir.exists():
            # Count .nc files
            nc_files = list(year_dir.glob("*.nc"))
            print(f"   [OK] {year_dir.name}: {len(nc_files)} .nc files")
        else:
            print(f"   [ERROR] Missing: {year_dir.name}")
    
    print()
    
    # Check 2: Try loading ERA5 loader
    print("2. Testing ERA5Loader initialization...")
    try:
        era5_loader = ERA5Loader(data_dir=str(era5_base))
        print(f"   [OK] ERA5Loader initialized")
        print(f"   [OK] Data directory: {era5_loader.data_dir}")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize ERA5Loader: {e}")
        return
    
    print()
    
    # Check 3: Try loading a sample storm
    print("3. Testing ERA5 loading for a sample storm...")
    try:
        ibtracs_loader = IBTrACSLoader(data_dir="data/raw")
        
        # Load interpolated tracks
        interpolated_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021.csv")
        if interpolated_file.exists():
            df = pd.read_csv(interpolated_file, low_memory=False)
            # Map columns
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
                df['USA_WIND'] = df['wind'] / 0.514444  # Convert m/s to knots for compatibility
            if 'pressure' in df.columns:
                df['WMO_PRES'] = df['pressure']
                df['USA_PRES'] = df['pressure']
            if 'ISO_TIME' in df.columns:
                df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
            elif 'time' in df.columns:
                df['ISO_TIME'] = pd.to_datetime(df['time'], errors='coerce')
            elif 'datetime' in df.columns:
                df['ISO_TIME'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            print(f"   [OK] Loaded {len(df)} interpolated records")
        else:
            print(f"   [ERROR] Interpolated tracks file not found: {interpolated_file}")
            return
        
        # Get a sample storm
        storm_ids = ibtracs_loader.filter_typhoons(
            df,
            start_year=2018,
            end_year=2021,
            min_wind_speed=33.0,
            min_duration_hours=48
        )
        
        if len(storm_ids) == 0:
            print("   [ERROR] No storms found")
            return
        
        test_storm_id = storm_ids[0]
        print(f"   Testing with storm: {test_storm_id}")
        
        storm_data = ibtracs_loader.get_storm_data(df, test_storm_id)
        print(f"   Storm name: {storm_data.get('name', 'Unknown')}")
        print(f"   Storm duration: {len(storm_data['times'])} timesteps")
        
        # Try loading ERA5 for this storm
        start_time = pd.to_datetime(storm_data['times'][0]) - pd.Timedelta(hours=6)
        end_time = pd.to_datetime(storm_data['times'][-1]) + pd.Timedelta(hours=6)
        
        lats = storm_data['lats']
        lons = storm_data['lons']
        lat_range = (float(min(lats) - 10), float(max(lats) + 10))
        lon_range = (float(min(lons) - 10), float(max(lons) + 10))
        
        print(f"   Time range: {start_time} to {end_time}")
        print(f"   Lat range: {lat_range}")
        print(f"   Lon range: {lon_range}")
        
        era5_ds = era5_loader.load_era5_from_daily_files(
            start_time=start_time,
            end_time=end_time,
            lat_range=lat_range,
            lon_range=lon_range
        )
        
        if era5_ds is None:
            print(f"   [ERROR] Failed to load ERA5 data for storm {test_storm_id}")
            print(f"   This means ERA5 extraction will fail for this storm")
        else:
            print(f"   [OK] Successfully loaded ERA5 dataset")
            print(f"   Dataset variables: {list(era5_ds.data_vars.keys())[:10]}...")
            print(f"   Dataset dimensions: {dict(era5_ds.dims)}")
            
            # Try extracting a frame
            print()
            print("4. Testing frame extraction...")
            try:
                test_lons = storm_data['lons'][:8]
                test_lats = storm_data['lats'][:8]
                test_times = storm_data['times'][:8]
                
                frames = era5_loader.extract_frames_at_times(
                    era5_ds, test_lons, test_lats, test_times, crop_size=64
                )
                
                if frames is None:
                    print(f"   [ERROR] Frame extraction returned None")
                elif frames.size == 0:
                    print(f"   [ERROR] Frame extraction returned empty array")
                else:
                    non_zero = (frames != 0).sum() / frames.size * 100
                    print(f"   [OK] Extracted frames: shape={frames.shape}")
                    print(f"   [OK] Non-zero data: {non_zero:.2f}%")
                    print(f"   [OK] Data range: [{frames.min():.4f}, {frames.max():.4f}]")
                    if non_zero < 1.0:
                        print(f"   [WARNING] Very few non-zero values - extraction may have issues")
                    else:
                        print(f"   [OK] Frame extraction working correctly!")
            except Exception as e:
                print(f"   [ERROR] Frame extraction failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"   [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

