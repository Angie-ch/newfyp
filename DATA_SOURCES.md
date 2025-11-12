# Typhoon Prediction Data Sources

## Overview

This project uses **IBTrACS WP** (Western Pacific typhoon tracks) and **ERA5** (meteorological reanalysis) for typhoon prediction.

## Data Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TYPHOON PREDICTION                        │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼─────┐            ┌───────▼─────┐
        │ IBTrACS WP  │            │    ERA5     │
        │   Tracks    │            │  Reanalysis │
        └─────────────┘            └─────────────┘
              │                           │
              │                           │
         Position                   Meteorology
         Intensity                  48 Channels
         Pressure                   25 km res.
              │                           │
              └──────────┬────────────────┘
                         │
                  ┌──────▼──────┐
                  │   Training  │
                  │   Samples   │
                  └─────────────┘
```

## 1. IBTrACS WP (Track Data)

### Description
International Best Track Archive for Climate Stewardship - Western Pacific Basin

### Coverage
- **Region**: Western Pacific Ocean only (120°E - 180°E, 0° - 60°N)
- **Time**: 1842 - present (best quality: 1980+)
- **Frequency**: 6-hourly observations
- **Basin Code**: WP

### Variables
| Variable | Unit | Source | Description |
|----------|------|--------|-------------|
| Latitude | degrees N | USA_LAT | Storm center latitude |
| Longitude | degrees E | USA_LON | Storm center longitude |
| Wind Speed | m/s | USA_WIND (converted from knots) | Maximum sustained wind |
| Pressure | hPa | USA_PRES | Minimum central pressure |

### Data Quality
- **Position Accuracy**: ±10-50 km (improves over time)
- **Intensity Accuracy**: ±5 m/s (typical)
- **Completeness**: High for major typhoons, lower for weak systems
- **Best Period**: 1980-present (satellite era)

### Access
- **Method**: Automatic HTTP download
- **URL**: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/
- **Format**: CSV
- **Size**: ~50 MB (entire WP basin)
- **Cache**: `data/raw/ibtracs_wp.csv`

### No Setup Required ✓
The IBTrACS data is automatically downloaded when first needed.

---

## 2. ERA5 (Meteorological Data)

### Description
ECMWF Reanalysis v5 - Global atmospheric reanalysis

### Coverage
- **Region**: Global (cropped to typhoon region)
- **Time**: 1950 - present (2-3 month delay)
- **Resolution**: 0.25° × 0.25° (~25 km at equator)
- **Frequency**: Hourly (we use 6-hourly)
- **Levels**: 7 pressure levels (200, 300, 500, 700, 850, 925, 1000 hPa)

### Variables (48 channels total)

#### Single-Level Variables (6 channels)
| Variable | Short Name | Unit | Description |
|----------|------------|------|-------------|
| Mean Sea Level Pressure | msl | Pa | Surface pressure |
| 10m U-wind | u10 | m/s | Eastward wind at 10m |
| 10m V-wind | v10 | m/s | Northward wind at 10m |
| 2m Temperature | t2m | K | Air temperature at 2m |
| Total Precipitation | tp | m | Accumulated precipitation |
| Total Column Water Vapor | tcwv | kg/m² | Atmospheric moisture |

#### Pressure-Level Variables (42 channels = 6 variables × 7 levels)
| Variable | Short Name | Unit | Levels |
|----------|------------|------|--------|
| Geopotential | z | m²/s² | 200-1000 hPa |
| Temperature | t | K | 200-1000 hPa |
| U-wind | u | m/s | 200-1000 hPa |
| V-wind | v | m/s | 200-1000 hPa |
| Relative Humidity | r | % | 200-1000 hPa |
| Vertical Velocity | w | Pa/s | 200-1000 hPa |

### Data Quality
- **Accuracy**: Comparable to or better than real-time analyses
- **Consistency**: Uniform across time and space
- **Limitations**:
  - 25 km resolution may miss fine-scale features
  - Peak intensity may be slightly underestimated for strong typhoons
  - 6-hour temporal resolution limits rapid intensification capture

### Access
- **Method**: CDS API (Copernicus Climate Data Store)
- **Authentication**: Required (free registration)
- **Format**: NetCDF
- **Size**: ~200 MB - 1 GB per storm
- **Cache**: `data/era5/{storm_id}_era5.nc`

### Setup Required
See `ERA5_SETUP.md` for detailed setup instructions.

**Quick Setup:**
1. Register at https://cds.climate.copernicus.eu/
2. Get API key from account settings
3. Install: `pip install cdsapi`
4. Configure: Create `~/.cdsapirc` with credentials

---

## 3. Output Format

### Training Samples

Each sample contains:

```python
{
    'past_frames': (T_past, C, H, W),      # e.g., (12, 48, 64, 64)
    'future_frames': (T_future, C, H, W),  # e.g., (8, 48, 64, 64)
    'past_track': (T_past, 2),             # [lon, lat]
    'future_track': (T_future, 2),         # [lon, lat]
    'past_intensity': (T_past,),           # wind speed (m/s)
    'future_intensity': (T_future,),       # wind speed (m/s)
    'past_pressure': (T_past,),            # pressure (hPa)
    'future_pressure': (T_future,),        # pressure (hPa)
    'storm_id': str,                       # e.g., "2020246N11154"
    'storm_name': str                      # e.g., "HAISHEN"
}
```

### Dimensions
- `T_past`: 12 timesteps (72 hours)
- `T_future`: 8 timesteps (48 hours)
- `C`: 48 channels (ERA5) or 40 channels (synthetic)
- `H, W`: 64 × 64 pixels (~16° × 16° at 0.25° resolution)

---

## 4. Comparison: ERA5 vs Synthetic

| Aspect | ERA5 | Synthetic |
|--------|------|-----------|
| **Data Source** | Real reanalysis | Generated from track |
| **Channels** | 48 physical variables | 40 synthetic patterns |
| **Setup** | CDS API required | None |
| **Download** | ~200 MB - 1 GB per storm | N/A |
| **Physics** | Realistic atmospheric state | Simplified patterns |
| **Quality** | High (research-grade) | Low (placeholder) |
| **Performance** | Better predictions expected | Baseline only |

### When to Use Each

**Use ERA5 if:**
- Training production models
- Need best prediction accuracy
- Have CDS API access
- Can wait for downloads

**Use Synthetic if:**
- Quick testing/debugging
- No internet access
- CDS API not set up
- Rapid iteration needed

---

## 5. Data Statistics (2018-2023 WP Typhoons)

### IBTrACS WP
- **Total storms**: ~180
- **Strong typhoons** (>17 m/s): ~120
- **Super typhoons** (>50 m/s): ~40
- **Average duration**: 5-10 days
- **Average track points**: 20-40

### ERA5 Coverage
- **Spatial extent per storm**: ~30° × 30° box
- **Temporal resolution**: 6 hours
- **File size per storm**: 200-1000 MB
- **Variables per timestep**: 48 channels

---

## 6. Quality Control

### IBTrACS Filtering
```python
# Default filters in IBTrACSLoader.filter_typhoons()
min_wind_speed = 17.0  # m/s (Tropical Storm threshold)
min_duration_hours = 24  # At least 1 day
min_records = 8  # At least 8 observations
```

### ERA5 Validation
- Check for NaN values in critical variables
- Verify spatial coverage around typhoon
- Ensure temporal alignment with IBTrACS
- Validate physical consistency (e.g., wind-pressure relationship)

---

## 7. Storage Requirements

### Minimal (IBTrACS only)
```
data/
├── raw/
│   └── ibtracs_wp.csv                    (~50 MB)
└── processed/
    └── typhoons_wp_synthetic.npz         (~100 MB for 100 samples)
```

### Full (IBTrACS + ERA5)
```
data/
├── raw/
│   └── ibtracs_wp.csv                    (~50 MB)
├── era5/
│   ├── 2020246N11154_era5.nc            (~500 MB)
│   ├── 2021257N14147_era5.nc            (~400 MB)
│   └── ... (one file per storm)
└── processed/
    └── typhoons_wp_era5.npz              (~500 MB for 100 samples)
```

**Recommendation**: Start with 5-10 storms for testing, expand as needed.

---

## 8. Usage Examples

### Quick Start (No ERA5)
```bash
python data/real_data_loader.py --demo
```

### With ERA5 (Using Cached Data)
```bash
python data/real_data_loader.py --use-era5
```

### Download ERA5 (First Time)
```bash
# Make sure ERA5 is set up first (see ERA5_SETUP.md)
python data/real_data_loader.py --use-era5 --download-era5
```

### Test Integration
```bash
python test_era5_integration.py
```

---

## 9. Data Citation

### IBTrACS
> Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010:
> The International Best Track Archive for Climate Stewardship (IBTrACS):
> Unifying tropical cyclone best track data. Bulletin of the American Meteorological Society, 91, 363-376.
> https://doi.org/10.1175/2009BAMS2755.1

### ERA5
> Hersbach, H., Bell, B., Berrisford, P., et al., 2020:
> The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146, 1999-2049.
> https://doi.org/10.1002/qj.3803

---

## 10. Support & Resources

### IBTrACS
- **Documentation**: https://www.ncei.noaa.gov/products/international-best-track-archive
- **Data Access**: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/
- **Support**: https://www.ncei.noaa.gov/support

### ERA5
- **Documentation**: https://confluence.ecmwf.int/display/CKB/ERA5
- **Data Access**: https://cds.climate.copernicus.eu/
- **Support**: https://support.ecmwf.int/
- **Setup Guide**: See `ERA5_SETUP.md` in this repository

### Code
- **Real Data Loader**: `data/real_data_loader.py`
- **Visualization**: `README_VISUALIZATION.md`
- **ERA5 Setup**: `ERA5_SETUP.md`
- **Integration Test**: `test_era5_integration.py`

