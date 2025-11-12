# ERA5 Data Integration Guide

This guide explains how to integrate ERA5 reanalysis data for typhoon prediction using IBTrACS WP (Western Pacific) tracks.

## Overview

The system now uses:
- **IBTrACS WP**: Western Pacific typhoon tracks and intensity data
- **ERA5**: ECMWF meteorological reanalysis data (optional but recommended)

## Data Sources

### 1. IBTrACS WP (Automatic)
- **Source**: NOAA International Best Track Archive for Climate Stewardship
- **Coverage**: Western Pacific typhoons only
- **Variables**: Position (lat/lon), wind speed, pressure
- **Download**: Automatic (no setup required)
- **URL**: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/

### 2. ERA5 (Requires Setup)
- **Source**: ECMWF Copernicus Climate Data Store
- **Coverage**: Global meteorological reanalysis (1950-present)
- **Resolution**: 0.25° × 0.25° (~25 km), 6-hourly
- **Variables**: 
  - Single-level: MSLP, 10m wind (u,v), 2m temperature, precipitation, water vapor
  - Pressure levels (200-1000 hPa): Geopotential, temperature, wind (u,v), humidity, vertical velocity
- **Total Channels**: 48 (6 single-level + 6 multi-level × 7 levels)

## ERA5 Setup (One-Time)

### Step 1: Register for CDS Access

1. Go to https://cds.climate.copernicus.eu/
2. Click "Register" and create a free account
3. Verify your email address

### Step 2: Get Your API Key

1. Log in to your account
2. Click your username (top right) → "Settings"
3. Scroll down to find your UID and API key
4. Copy both values

### Step 3: Install CDS API Client

```bash
pip install cdsapi xarray scipy netCDF4
```

### Step 4: Configure API Key

Create a file `~/.cdsapirc` with your credentials:

```bash
# On Mac/Linux
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF

# Set correct permissions
chmod 600 ~/.cdsapirc
```

Replace `YOUR_UID:YOUR_API_KEY` with your actual values from Step 2.

Example:
```
url: https://cds.climate.copernicus.eu/api/v2
key: 12345:abcdef01-2345-6789-abcd-ef0123456789
```

### Step 5: Accept Terms of Use

Before downloading, you must accept the license:

1. Go to https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
2. Read and accept the terms
3. Go to ERA5 dataset pages and accept their specific terms:
   - Single-level: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
   - Pressure-level: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels

## Usage

### Option 1: Quick Demo (No ERA5 Required)

Uses IBTrACS WP tracks with synthetic meteorological frames:

```bash
cd /Volumes/data/fyp/typhoon_prediction
python data/real_data_loader.py --demo
```

### Option 2: With ERA5 Data (Cached)

Uses pre-downloaded ERA5 data if available:

```bash
python data/real_data_loader.py --use-era5
```

### Option 3: Download ERA5 (Full Pipeline)

Downloads ERA5 data for storms (requires CDS API setup):

```bash
python data/real_data_loader.py --use-era5 --download-era5
```

**Note**: ERA5 downloads can be slow (several GB per storm). The system caches data automatically.

### Full Training Pipeline with ERA5

```python
from data.real_data_loader import IBTrACSLoader

# Create loader
loader = IBTrACSLoader(data_dir="data/raw")

# Create dataset with ERA5
samples = loader.create_dataset(
    n_samples=100,
    start_year=2018,
    end_year=2023,
    use_era5=True,           # Use ERA5 data
    download_era5=True,      # Download if needed
    save_path="data/processed/typhoons_wp_era5.npz"
)

print(f"Created {len(samples)} samples")
print(f"Frame shape: {samples[0]['past_frames'].shape}")  # (T, 48, 64, 64)
```

## Data Specifications

### IBTrACS WP Output
- **Track data**: `(T, 2)` - longitude, latitude
- **Intensity**: `(T,)` - wind speed (m/s)
- **Pressure**: `(T,)` - central pressure (hPa)
- **Temporal resolution**: 6 hours
- **Basin**: Western Pacific only

### ERA5 Channels (48 total)

#### Single-Level Variables (6 channels)
1. Mean sea level pressure (MSLP)
2. 10m u-component of wind
3. 10m v-component of wind
4. 2m temperature
5. Total precipitation
6. Total column water vapor

#### Pressure-Level Variables (42 channels = 6 vars × 7 levels)

Variables:
1. Geopotential height
2. Temperature
3. U-component of wind
4. V-component of wind
5. Relative humidity
6. Vertical velocity (omega)

Pressure levels (hPa):
- 200, 300, 500, 700, 850, 925, 1000

### Output Format
- **Shape**: `(T, 48, 64, 64)` where:
  - `T` = number of timesteps (12 past + 8 future)
  - `48` = number of ERA5 channels
  - `64×64` = spatial grid (pixels)
- **Spatial coverage**: ~16° × 16° around typhoon center
- **Format**: NumPy arrays (float32)

## Fallback Behavior

The system gracefully handles missing ERA5 data:

1. **If ERA5 not configured**: Uses synthetic frames (pattern-based)
2. **If ERA5 download fails**: Falls back to synthetic frames with warning
3. **If ERA5 unavailable for specific storm**: Uses synthetic frames for that storm only

This ensures the pipeline always works, even without ERA5 setup.

## Performance Considerations

### Storage Requirements
- **IBTrACS**: ~50 MB (entire Western Pacific database)
- **ERA5 per storm**: 
  - Small storm (3 days): ~200 MB
  - Large storm (10 days): ~1 GB
  - Cached storms reused automatically

### Download Times
- **IBTrACS**: ~10 seconds (one-time)
- **ERA5 per storm**: 5-30 minutes depending on:
  - Storm duration
  - CDS queue length
  - Network speed

### Recommendations
1. Start with `--demo` mode for testing
2. Download ERA5 for 5-10 key storms first
3. Use cached data for repeated experiments
4. Consider pre-downloading during off-peak hours

## Troubleshooting

### Error: "cdsapi not installed"
```bash
pip install cdsapi xarray scipy netCDF4
```

### Error: "Client not authorized"
- Check `~/.cdsapirc` file exists and has correct format
- Verify UID and API key are correct
- Ensure you've accepted the CDS terms of use

### Error: "Request failed"
- Check CDS system status: https://cds.climate.copernicus.eu/
- CDS may have maintenance periods
- Try again later or use cached data

### Downloads are very slow
- CDS can have queues during peak hours
- Consider downloading overnight
- Use `--use-era5` without `--download-era5` to use cached data only

### "No storms found matching criteria"
- Adjust year range: `start_year` and `end_year`
- Lower `min_wind_speed` threshold
- Check IBTrACS data was downloaded correctly

## Example: Complete Workflow

```bash
# 1. Quick test (no ERA5 needed)
python data/real_data_loader.py --demo

# 2. Set up ERA5 credentials (one-time)
nano ~/.cdsapirc  # Add your credentials

# 3. Download ERA5 for a few storms
python data/real_data_loader.py --use-era5 --download-era5

# 4. Train model with ERA5 data
python test_with_real_data.py

# 5. Visualize results
python demo_visualizations.py
open demo_visualizations/prediction_report.html
```

## Data Quality Notes

### IBTrACS WP
- **Quality**: Operational data from multiple agencies
- **Accuracy**: Position ±10-50 km, intensity ±5 m/s (typical)
- **Coverage**: Complete for WP basin since 1980s
- **Missing data**: Occasional gaps, especially for weak systems

### ERA5
- **Quality**: State-of-the-art reanalysis (model + observations)
- **Accuracy**: Better than real-time forecasts for historical events
- **Consistency**: Uniform quality across time and space
- **Limitations**: 
  - 25 km resolution may miss small-scale features
  - 6-hour temporal resolution
  - Slight underestimation of peak typhoon intensity

## References

- **IBTrACS**: Knapp et al. (2010), https://doi.org/10.1175/2009BAMS2755.1
- **ERA5**: Hersbach et al. (2020), https://doi.org/10.1002/qj.3803
- **CDS API**: https://cds.climate.copernicus.eu/api-how-to

## Support

- IBTrACS issues: https://www.ncei.noaa.gov/support
- ERA5/CDS issues: https://support.ecmwf.int/
- Code issues: Check project README.md

