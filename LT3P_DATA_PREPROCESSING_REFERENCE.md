# LT3P Data Preprocessing Methodology Reference

## Reference
**LT3P Repository**: https://github.com/iclr2024submit/LT3P  
**Paper**: "Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data" (ICLR 2024)

## LT3P Data Sources

According to the LT3P paper and repository, they use:
1. **UM (Unified Model) Forecast Data** - Real-time operational weather forecasts
2. **Best Track Data** - Ground truth typhoon tracks (similar to IBTrACS)
3. **ERA5 Reanalysis Data** - For comparison/evaluation (they mention releasing preprocessed ERA5 data)

## LT3P Data Preprocessing Approach

### Key Characteristics (Inferred from Paper)
1. **Temporal Resolution**: 6-hour intervals (matches our approach)
2. **Spatial Cropping**: Region around typhoon center (matches our approach)
3. **Multi-Source Integration**: Combines UM forecasts with best-track data
4. **Preprocessed Dataset**: They release a "PHYSICS TRACK dataset" with preprocessed data

### Folder Structure (from LT3P repo)
```
./UM_2019/          # UM forecast data by year
    .npy files      # Preprocessed numpy arrays
./2019/             # Best track data by year
    .txt files      # Text files with track information
./1900.pth          # Pre-trained model weights
```

## Our Current Approach vs. LT3P

### Similarities
✅ **Temporal Resolution**: Both use 6-hour intervals  
✅ **Spatial Cropping**: Both crop regions around typhoon center  
✅ **Multi-Task Data**: Both include atmospheric fields + track + intensity  
✅ **Preprocessing Pipeline**: Both preprocess data into training-ready format

### Differences

| Aspect | LT3P | Our Approach |
|--------|------|--------------|
| **Primary Data** | UM forecast (real-time) | ERA5 reanalysis (historical) |
| **Track Data** | Best track (text files) | IBTrACS (CSV format) |
| **File Format** | `.npy` arrays | `.npz` compressed |
| **Organization** | By year directories | By split (train/val/test) |
| **Temporal Split** | Not specified | Yes (2018-2019/2020/2021) |

## Recommended Improvements Based on LT3P

### 1. Data Organization Structure

**LT3P Structure:**
```
data/
  UM_2018/
    storm_001.npy
    storm_002.npy
  UM_2019/
    ...
  2018/
    track_001.txt
    track_002.txt
```

**Our Current Structure:**
```
data/
  processed_temporal_split/
    train/cases/*.npz
    val/cases/*.npz
    test/cases/*.npz
```

**Recommendation**: Our structure is better for ML training (already split), but we could add:
- Year-based organization for easier data management
- Separate track files for reference

### 2. Data Preprocessing Steps

Based on LT3P methodology, here's how we should structure our preprocessing:

#### Step 1: Load Track Data (IBTrACS)
```python
# Similar to LT3P's best track loading
# Our implementation: data/real_data_loader.py - IBTrACSLoader
- Load IBTrACS CSV files
- Filter for Western Pacific typhoons
- Extract track positions, intensity, pressure
- Interpolate to 6-hour intervals
```

#### Step 2: Load Atmospheric Data (ERA5)
```python
# Similar to LT3P's UM data loading
# Our implementation: data/real_data_loader.py - ERA5Loader
- Load ERA5 daily files
- Extract variables at multiple pressure levels
- Crop region around typhoon center
- Align with track timesteps
```

#### Step 3: Create Training Samples
```python
# Our implementation: data/generate_data_by_year.py
- Sliding window approach (past + future)
- Combine atmospheric fields + track + intensity
- Save as .npz files
- Split by year (temporal split)
```

### 3. Key Preprocessing Features from LT3P

#### A. Temporal Alignment
- **LT3P**: Aligns UM forecasts with best track timesteps
- **Our Approach**: Aligns ERA5 data with IBTrACS timesteps
- **Status**: ✅ Implemented in `create_training_sample()`

#### B. Spatial Cropping
- **LT3P**: Crops region around typhoon center
- **Our Approach**: Crops 64×64 region around center
- **Status**: ✅ Implemented in `extract_frames_at_times()`

#### C. Variable Selection
- **LT3P**: Uses UM forecast variables
- **Our Approach**: Uses ERA5 variables (7 vars × 4 levels = 28 channels)
- **Status**: ✅ Implemented with configurable variables

#### D. Normalization
- **LT3P**: Likely normalizes UM data
- **Our Approach**: Normalizes ERA5 data using global statistics
- **Status**: ✅ Implemented in `TyphoonDataset`

## Implementation Recommendations

### 1. Add Data Validation Step (Like LT3P)
```python
# Add to data/generate_data_by_year.py
def validate_sample_quality(sample):
    """
    Validate sample quality similar to LT3P preprocessing
    - Check for missing data
    - Verify temporal continuity
    - Validate spatial coverage
    """
    # Check atmospheric fields
    if np.any(np.isnan(sample['past_frames'])):
        return False
    
    # Check track continuity
    track_diff = np.diff(sample['past_track'], axis=0)
    if np.any(np.abs(track_diff) > 10.0):  # Unrealistic jumps
        return False
    
    # Check intensity consistency
    if np.any(sample['past_intensity'] < 0):
        return False
    
    return True
```

### 2. Add Data Statistics (Like LT3P Dataset)
```python
# Add comprehensive statistics
def compute_dataset_statistics(samples):
    """
    Compute statistics similar to LT3P's preprocessed dataset
    """
    stats = {
        'n_samples': len(samples),
        'n_storms': len(set(s['storm_id'] for s in samples)),
        'temporal_coverage': {
            'min_date': min(s['times'][0] for s in samples),
            'max_date': max(s['times'][-1] for s in samples)
        },
        'spatial_coverage': {
            'lat_range': (min, max),
            'lon_range': (min, max)
        },
        'intensity_stats': {
            'mean': np.mean([s['past_intensity'].mean() for s in samples]),
            'std': np.std([s['past_intensity'].mean() for s in samples]),
            'max': np.max([s['past_intensity'].max() for s in samples])
        }
    }
    return stats
```

### 3. Improve Data Loading Efficiency
```python
# Optimize similar to LT3P's .npy format
# Consider using .npy for faster loading if needed
def save_as_npy(sample, output_path):
    """
    Save in .npy format for faster loading (like LT3P)
    """
    # Save each component separately
    np.save(f"{output_path}_frames.npy", sample['past_frames'])
    np.save(f"{output_path}_track.npy", sample['past_track'])
    np.save(f"{output_path}_intensity.npy", sample['past_intensity'])
```

### 4. Add Metadata Files (Like LT3P)
```python
# Create comprehensive metadata
metadata = {
    'dataset_name': 'PHYSICS_TRACK_ERA5_IBTrACS',
    'data_sources': {
        'atmospheric': 'ERA5 reanalysis',
        'track': 'IBTrACS Western Pacific',
        'version': 'v1.0'
    },
    'preprocessing': {
        'temporal_resolution': '6-hour',
        'spatial_resolution': '0.25° (ERA5 native)',
        'crop_size': '64×64 pixels',
        'normalization': 'global statistics'
    },
    'splits': {
        'train': {'years': [2018, 2019], 'n_samples': N},
        'val': {'years': [2020], 'n_samples': M},
        'test': {'years': [2021], 'n_samples': K}
    },
    'variables': {
        'atmospheric': ['z', 't', 'u', 'v', 'r', 'q', 'vo'],
        'pressure_levels': [1000, 850, 500, 250],
        'track': ['lat', 'lon'],
        'intensity': ['wind_speed', 'pressure']
    }
}
```

## Current Implementation Status

### ✅ Already Implemented (Similar to LT3P)
- [x] Temporal alignment (6-hour intervals)
- [x] Spatial cropping around typhoon center
- [x] Multi-variable atmospheric data (ERA5)
- [x] Track and intensity data (IBTrACS)
- [x] Sliding window sample generation
- [x] Temporal split (train/val/test by year)
- [x] Data normalization

### 🔄 Could Be Improved (Based on LT3P)
- [ ] Add data quality validation step
- [ ] Add comprehensive dataset statistics
- [ ] Consider .npy format for faster loading
- [ ] Add detailed metadata files
- [ ] Add data versioning

## Data Regeneration Process (Based on LT3P Methodology)

### Recommended Workflow

```python
# 1. Load and validate track data
ibtracs_loader = IBTrACSLoader()
df = ibtracs_loader.load_ibtracs()
storm_ids = ibtracs_loader.filter_typhoons(df, start_year=2018, end_year=2021)

# 2. Load ERA5 data for each storm
era5_loader = ERA5Loader()
era5_datasets = {}
for storm_id in storm_ids:
    era5_ds = era5_loader.load_era5_from_daily_files(...)
    if era5_ds is not None:
        era5_datasets[storm_id] = era5_ds

# 3. Generate samples with validation
samples = []
for storm_id in storm_ids:
    storm_data = ibtracs_loader.get_storm_data(df, storm_id)
    era5_ds = era5_datasets.get(storm_id)
    
    # Create samples with sliding window
    for start_idx in range(0, len(storm_data['times']) - total_timesteps, stride):
        sample = create_training_sample(
            storm_data, era5_dataset=era5_ds, ...
        )
        
        # Validate sample quality (like LT3P)
        if validate_sample_quality(sample):
            samples.append(sample)

# 4. Split by year (temporal split)
train_samples = [s for s in samples if s['year'] in [2018, 2019]]
val_samples = [s for s in samples if s['year'] == 2020]
test_samples = [s for s in samples if s['year'] == 2021]

# 5. Save with metadata
save_samples_by_split(train_samples, val_samples, test_samples, metadata)
```

## Key Takeaways from LT3P

1. **Preprocessing is Critical**: LT3P emphasizes careful preprocessing and validation
2. **Multi-Source Integration**: Combine atmospheric data with track data carefully
3. **Temporal Alignment**: Ensure perfect alignment between data sources
4. **Quality Control**: Validate samples before including in dataset
5. **Comprehensive Metadata**: Document all preprocessing steps and data sources

## References

- LT3P Repository: https://github.com/iclr2024submit/LT3P
- LT3P Paper: Park et al., ICLR 2024 Spotlight
- ERA5 Documentation: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- IBTrACS Documentation: https://www.ncei.noaa.gov/products/international-best-track-archive


