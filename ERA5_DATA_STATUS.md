# ERA5 Data Status Report

## ✓ Data Validation Complete

Date: November 7, 2025

### Summary

Your ERA5 data has been **verified and is ready to use** with the typhoon prediction pipeline!

## Data Inventory

### Files Available
- **2018**: 216 pressure-level files
- **2019**: 220 pressure-level files  
- **2020**: 146 pressure-level files
- **2021**: 205 pressure-level files
- **Total**: 787 ERA5 pressure-level (era5_pl_*.nc) files

### Data Format

All files have been validated to have the correct structure:

```
Dimensions:
  - valid_time: 4 timesteps (6-hour intervals)
  - pressure_level: 4 levels [1000, 850, 500, 250] hPa
  - latitude: ~40-45 points
  - longitude: ~42-49 points

Variables (7 atmospheric fields):
  - z: Geopotential
  - r: Relative humidity
  - q: Specific humidity
  - t: Temperature
  - u: U component of wind
  - v: V component of wind
  - vo: Vorticity (relative)
```

### Data Quality
- ✓ No NaN values detected
- ✓ No Inf values detected
- ✓ All files have consistent structure
- ✓ Regular 6-hour time spacing
- ✓ Appropriate spatial coverage for Western Pacific typhoons

## Pipeline Configuration

### Current Setup: 8 Past + 12 Future

With your ERA5 data at 6-hour intervals:

```
Input:  8 timesteps × 6 hours = 48 hours (2 days) of history
Output: 12 timesteps × 6 hours = 72 hours (3 days) of prediction
Total:  120 hours (5 days) per training sample
```

### Sample Structure

Each generated sample will have:
```python
{
    'past_frames': (8, 48, 64, 64),      # 8 historical timesteps
    'future_frames': (12, 48, 64, 64),   # 12 future timesteps to predict
    'track_past': (8, 2),                # Past (lat, lon) positions
    'track_future': (12, 2),             # Future positions to predict
    'intensity_past': (8,),              # Past wind speeds
    'intensity_future': (12,)            # Future wind speeds to predict
}
```

Where 48 channels = 7 variables × 4 pressure levels + other features.

## Next Steps

### 1. Generate Training Samples

Run the pipeline to create 100 training samples:

```bash
python run_real_data_pipeline.py --n-samples 100
```

This will:
- Load IBTrACS typhoon tracks
- Match them with ERA5 meteorological data
- Create samples with 8 past + 12 future timesteps
- Save to `data/processed/cases/`

### 2. Expected Output

The pipeline will generate:
```
data/processed/
├── cases/
│   ├── case_0000.npz
│   ├── case_0001.npz
│   ├── ...
│   └── case_0099.npz
├── statistics.json
└── metadata.json
```

### 3. Training

After generating samples, the pipeline will automatically:
1. Train the spatial autoencoder
2. Train the physics-informed diffusion model
3. Evaluate on test set
4. Generate visualizations

## Environment Setup

The following packages were installed in the `ai` conda environment:
- ✓ netCDF4 (for reading ERA5 files)
- ✓ h5netcdf (alternative netCDF backend)
- ✓ xarray (for data manipulation)

To use the pipeline, make sure you're in the correct environment:
```bash
conda activate ai
cd /Volumes/data/fyp/typhoon_prediction
python run_real_data_pipeline.py --n-samples 100
```

## File Coverage by Year

### 2018 (216 files)
- Date range: 2018-02-08 to 2018-12-31
- Spatial coverage verified across Western Pacific

### 2019 (220 files)  
- Date range: 2019-01-01 to 2019-12-29
- Includes several major typhoons

### 2020 (146 files)
- Date range: 2020-05-08 to 2020-12-25
- Partial year coverage

### 2021 (205 files)
- Date range: 2021-02-14 to 2021-12-21
- Good coverage of typhoon season

## Notes

### Surface Level Files Ignored
- Your dataset also contains `era5_sl_*.nc` files (single-level data)
- These are **not used** by the current pipeline
- Only pressure-level files (`era5_pl_*.nc`) are needed
- This is expected behavior

### Missing cdsapi Warning
- You may see a warning about `cdsapi not installed`
- This is **not a problem** since you already have ERA5 data
- `cdsapi` is only needed to download new ERA5 data from Copernicus

## Validation Tests Run

1. ✓ File structure validation (dimensions, variables)
2. ✓ Data quality check (NaN, Inf values)
3. ✓ Time spacing verification (6-hour intervals)
4. ✓ Spatial coverage check
5. ✓ Pipeline loader compatibility test

All tests passed successfully!

## Support Files

The following test scripts were created for validation:
- `test_era5_simple.py` - Quick ERA5 format validation
- `test_era5_data.py` - Comprehensive data testing

You can re-run these anytime to verify data integrity:
```bash
python test_era5_simple.py
```

