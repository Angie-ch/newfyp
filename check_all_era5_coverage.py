"""
Check spatial coverage of all ERA5 files to determine if they are pre-cropped
"""
import xarray as xr
from pathlib import Path
import numpy as np

print("="*80)
print("CHECKING ERA5 FILE SPATIAL COVERAGE")
print("="*80)
print()

# Check files from different years
years = [2018, 2019, 2020, 2021]
all_lon_ranges = []
all_lat_ranges = []

for year in years:
    year_dir = Path(f"data/era5/ERA5_{year}_26data")
    if not year_dir.exists():
        print(f"Year {year}: Directory not found")
        continue
    
    files = list(year_dir.glob("era5_pl_*.nc"))
    if not files:
        print(f"Year {year}: No files found")
        continue
    
    # Check first few files
    print(f"Year {year} ({len(files)} files):")
    sample_files = files[:3]
    
    year_lon_ranges = []
    year_lat_ranges = []
    
    for f in sample_files:
        try:
            ds = xr.open_dataset(f)
            lon_min = float(ds.coords['longitude'].min().values)
            lon_max = float(ds.coords['longitude'].max().values)
            lat_min = float(ds.coords['latitude'].min().values)
            lat_max = float(ds.coords['latitude'].max().values)
            
            year_lon_ranges.append((lon_min, lon_max))
            year_lat_ranges.append((lat_min, lat_max))
            
            print(f"  {f.name}:")
            print(f"    Longitude: [{lon_min:.2f}, {lon_max:.2f}] ({lon_max-lon_min:.2f}°)")
            print(f"    Latitude: [{lat_min:.2f}, {lat_max:.2f}] ({lat_max-lat_min:.2f}°)")
            
            ds.close()
        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")
    
    if year_lon_ranges:
        # Check if all files have same coverage
        lon_min_all = min(r[0] for r in year_lon_ranges)
        lon_max_all = max(r[1] for r in year_lon_ranges)
        lat_min_all = min(r[0] for r in year_lat_ranges)
        lat_max_all = max(r[1] for r in year_lat_ranges)
        
        all_lon_ranges.append((lon_min_all, lon_max_all))
        all_lat_ranges.append((lat_min_all, lat_max_all))
        
        print(f"  Year range: lon=[{lon_min_all:.2f}, {lon_max_all:.2f}], lat=[{lat_min_all:.2f}, {lat_max_all:.2f}]")
    
    print()

# Overall coverage
if all_lon_ranges:
    overall_lon_min = min(r[0] for r in all_lon_ranges)
    overall_lon_max = max(r[1] for r in all_lon_ranges)
    overall_lat_min = min(r[0] for r in all_lat_ranges)
    overall_lat_max = max(r[1] for r in all_lat_ranges)
    
    print("="*80)
    print("OVERALL COVERAGE:")
    print(f"  Longitude: [{overall_lon_min:.2f}, {overall_lon_max:.2f}] ({overall_lon_max-overall_lon_min:.2f} degrees)")
    print(f"  Latitude: [{overall_lat_min:.2f}, {overall_lat_max:.2f}] ({overall_lat_max-overall_lat_min:.2f} degrees)")
    print()
    
    # Compare to expected Western Pacific coverage
    print("COMPARISON:")
    print("  Expected WP typhoon region:")
    print("    Longitude: 100°E to 180°E (80 degrees)")
    print("    Latitude: 5°N to 45°N (40 degrees)")
    print()
    print("  Your ERA5 files:")
    print(f"    Longitude: {overall_lon_min:.1f}°E to {overall_lon_max:.1f}°E ({overall_lon_max-overall_lon_min:.1f} degrees)")
    print(f"    Latitude: {overall_lat_min:.1f}°N to {overall_lat_max:.1f}°N ({overall_lat_max-overall_lat_min:.1f} degrees)")
    print()
    
    # Determine if cropped
    wp_lon_range = (100, 180)
    wp_lat_range = (5, 45)
    
    is_cropped = (
        overall_lon_min > wp_lon_range[0] or overall_lon_max < wp_lon_range[1] or
        overall_lat_min > wp_lat_range[0] or overall_lat_max < wp_lat_range[1]
    )
    
    if is_cropped:
        print("  [RESULT] YES - Files are PRE-CROPPED to a smaller region")
        print(f"  Coverage: {((overall_lon_max-overall_lon_min) * (overall_lat_max-overall_lat_min)) / (80 * 40) * 100:.1f}% of expected WP region")
    else:
        print("  [RESULT] NO - Files appear to have full Western Pacific coverage")










