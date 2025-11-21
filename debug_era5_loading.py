"""Debug ERA5 loading for a specific typhoon"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Load tracks
tracks_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021_6h.csv")
tracks_df = pd.read_csv(tracks_file)
tracks_df['ISO_TIME'] = pd.to_datetime(tracks_df['ISO_TIME'])

# Test with the March 2018 typhoon
typhoon_id = '2018082N04147'
storm = tracks_df[tracks_df['typhoon_id'] == typhoon_id].copy()
storm = storm.sort_values('ISO_TIME')

print("="*80)
print(f"TYPHOON {typhoon_id} - ERA5 FILE MATCHING")
print("="*80)

print(f"\nStorm period: {storm['ISO_TIME'].min()} to {storm['ISO_TIME'].max()}")
print(f"Total records: {len(storm)}")

# Check which ERA5 files should exist
start_date = storm['ISO_TIME'].min().date()
end_date = storm['ISO_TIME'].max().date()

print(f"\nRequired ERA5 files (from {start_date} to {end_date}):")
print("-"*80)

era5_dir = Path("data/era5/ERA5_2018_26data")
current_date = start_date

missing_files = []
existing_files = []

while current_date <= end_date:
    era5_filename = f"era5_pl_{current_date.strftime('%Y%m%d')}.nc"
    era5_path = era5_dir / era5_filename
    
    # Check which typhoon timesteps fall on this date
    timesteps_on_date = storm[storm['ISO_TIME'].dt.date == current_date]
    
    if era5_path.exists():
        status = "[EXISTS]"
        existing_files.append(era5_filename)
    else:
        status = "[MISSING]"
        missing_files.append(era5_filename)
    
    print(f"{status} {era5_filename} - {len(timesteps_on_date)} typhoon timesteps")
    if len(timesteps_on_date) > 0:
        for _, row in timesteps_on_date.iterrows():
            print(f"         {row['ISO_TIME']} - Lat: {row['lat']:.2f}, Lon: {row['lon']:.2f}")
    
    current_date += timedelta(days=1)

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"{'='*80}")
print(f"Existing ERA5 files: {len(existing_files)}")
print(f"Missing ERA5 files: {len(missing_files)}")

if len(existing_files) > 0:
    print(f"\nExisting files: {existing_files[:5]}...")
if len(missing_files) > 0:
    print(f"\nMissing files: {missing_files[:5]}...")

# Now check if existing files cover the typhoon locations
if len(existing_files) > 0:
    print(f"\n{'='*80}")
    print("CHECKING SPATIAL COVERAGE OF EXISTING FILES:")
    print(f"{'='*80}")
    
    import xarray as xr
    
    for filename in existing_files[:3]:  # Check first 3
        filepath = era5_dir / filename
        try:
            ds = xr.open_dataset(filepath, engine='h5netcdf')
            lat_min, lat_max = ds.coords['latitude'].values.min(), ds.coords['latitude'].values.max()
            lon_min, lon_max = ds.coords['longitude'].values.min(), ds.coords['longitude'].values.max()
            
            # Get typhoon positions on this date
            file_date = datetime.strptime(filename.split('_')[-1].replace('.nc', ''), '%Y%m%d').date()
            typhoon_positions = storm[storm['ISO_TIME'].dt.date == file_date]
            
            print(f"\n{filename}:")
            print(f"  ERA5 coverage: Lat [{lat_min:.2f}, {lat_max:.2f}], Lon [{lon_min:.2f}, {lon_max:.2f}]")
            
            for _, pos in typhoon_positions.iterrows():
                lat_ok = lat_min <= pos['lat'] <= lat_max
                lon_ok = lon_min <= pos['lon'] <= lon_max
                status = "[OK]" if (lat_ok and lon_ok) else "[OUT OF BOUNDS]"
                print(f"  {pos['ISO_TIME']}: Lat {pos['lat']:.2f}, Lon {pos['lon']:.2f} {status}")
            
            ds.close()
        except Exception as e:
            print(f"\n{filename}: [ERROR] {e}")

