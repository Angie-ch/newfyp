# ERA5 Coverage Issue - Root Cause and Fix

## Problem Summary

The dataset was showing "Synthetic" because ERA5 extraction was returning all zeros. The root cause was:

1. **Pre-cropped ERA5 files**: Your ERA5 files in `data/era5/ERA5_2018_26data/` are pre-cropped to a small region:
   - Longitude: 142.8°E to 154.8°E
   - Latitude: 4.5°N to 15.5°N

2. **Most typhoons are outside this region**: Western Pacific typhoons typically occur between:
   - Longitude: 100°E to 180°E
   - Latitude: 5°N to 45°N

3. **Empty spatial selection**: When trying to extract data for a typhoon at 130°E, the code requested longitude range [122°E, 138°E], but the file only has [142.8°E, 154.8°E]. The spatial selection returned an empty array (longitude: 0), resulting in all-zero frames.

## Fixes Applied

### 1. Improved Spatial Selection in `load_era5_from_daily_files`
- Added overlap checking: Only loads files that overlap with the requested region
- Handles pre-cropped files: If file is smaller than requested, uses it as-is
- Skips files that don't cover the requested region (no error, just skip)

### 2. Enhanced Extraction Function `extract_frames_at_times`
- Checks if storm center is within file coverage before extraction
- Checks if crop region overlaps with file coverage
- Provides clear error messages showing requested vs. available ranges
- Raises errors (which are caught) when storm is outside coverage, causing sample to be skipped

### 3. Better Error Handling
- Empty crops are detected and raise errors (causing sample to be skipped)
- Suppressed warnings for expected cases (pre-cropped files that don't cover region)

## Current Behavior

With these fixes:
1. **Files that don't cover typhoon location**: Skipped silently (no error)
2. **Typhoons outside file coverage**: Samples are skipped (no synthetic fallback)
3. **Typhoons within file coverage**: Data is extracted correctly

## What You Need to Do

### Option 1: Get Full-Coverage ERA5 Files (Recommended)

Download ERA5 data that covers the full Western Pacific region:
- Longitude: 100°E to 180°E (or wider)
- Latitude: 0°N to 50°N (or wider)

This will allow processing all typhoons in your dataset.

### Option 2: Use Current Pre-Cropped Files

With the current pre-cropped files, only typhoons within 142.8-154.8°E and 4.5-15.5°N will have data. Others will be skipped.

To check which typhoons are covered:
```python
# Check typhoon locations vs ERA5 file coverage
python check_era5_coverage.py
```

## Next Steps

1. **Regenerate dataset** with the fixes:
   ```bash
   python data/generate_data_by_year.py
   ```

2. **Check results**: The dataset will only include typhoons that have ERA5 data coverage. Samples for typhoons outside coverage will be skipped.

3. **Verify**: Check `dataset_info.json` - it should show "meteorological_data": "ERA5" for samples that were successfully extracted.

## Expected Outcome

- **Before fix**: All samples showed "Synthetic" because extraction returned zeros
- **After fix**: Samples within ERA5 coverage show "ERA5", samples outside coverage are skipped

The dataset will be smaller but will contain only real ERA5 data (no synthetic fallback).










