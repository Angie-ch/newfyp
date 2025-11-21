"""Check ERA5 variable names"""
import xarray as xr
from pathlib import Path

era5_dir = Path('data/era5/ERA5_2018_26data')
files = list(era5_dir.glob('era5_pl_*.nc'))
if files:
    ds = xr.open_dataset(files[0])
    print('ERA5 Variables:')
    print(list(ds.data_vars.keys()))
    print('\nDimensions:')
    print(dict(ds.dims))
    print('\nCoordinates:')
    print(list(ds.coords.keys()))
    print('\nSample variable structure:')
    if ds.data_vars:
        var_name = list(ds.data_vars.keys())[0]
        print(f'{var_name}: {ds[var_name].dims}, shape: {ds[var_name].shape}')











