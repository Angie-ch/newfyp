"""Check actual dimensions of ERA5 files"""
import xarray as xr
from pathlib import Path
import glob

files = sorted(glob.glob('data/era5/ERA5_2018_26data/era5_pl_*.nc'))[:10]

print('Checking ERA5 file dimensions:')
print('='*60)

dimensions = []
for f in files:
    try:
        ds = xr.open_dataset(f)
        lat_size = ds.sizes.get('latitude', 0)
        lon_size = ds.sizes.get('longitude', 0)
        dims = (lat_size, lon_size)
        dimensions.append(dims)
        print(f'{Path(f).name}:')
        print(f'  Grid size: {lat_size}x{lon_size}')
        ds.close()
    except Exception as e:
        print(f'{Path(f).name}: Error - {e}')

if dimensions:
    print('\n' + '='*60)
    print('Summary:')
    unique_dims = set(dimensions)
    print(f'Unique dimensions found: {unique_dims}')
    if len(unique_dims) == 1:
        dim = list(unique_dims)[0]
        print(f'\nRecommended crop_size: {min(dim)} (or smaller)')
        print(f'Current default: 64')
        if min(dim) < 64:
            print(f'⚠️  Warning: Files are smaller than 64x64!')
            print(f'   Should use crop_size={min(dim)} or smaller')


