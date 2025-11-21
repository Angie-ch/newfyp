"""Test ERA5 loading"""
from data.real_data_loader import ERA5Loader
import pandas as pd

loader = ERA5Loader()
print('Testing ERA5 loading...')
start = pd.to_datetime('2018-08-01')
end = pd.to_datetime('2018-08-05')
ds = loader.load_era5_from_daily_files(start, end, (0, 20), (100, 140))
print('Result:', 'Success' if ds is not None else 'Failed')
if ds is not None:
    print('Dataset variables:', list(ds.data_vars.keys())[:5])
    print('Dataset shape:', dict(ds.dims))











