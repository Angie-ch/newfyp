# Regeneration Started - ERA5 Only (No Synthetic)

## ✅ Status: RUNNING

**Process ID**: 17460  
**Started**: 2025-11-20 18:20:58  
**Log File**: `regeneration_log_20251120_182057.txt`

## Configuration Verified

### 1. ERA5 Only (No Synthetic) ✓
- **Code**: `data/generate_data_by_year.py`
  - Line 97: `"REQUIRED, no synthetic data allowed"`
  - Line 99-101: Storms without ERA5 are **SKIPPED** (not given synthetic data)
  - Line 104: `use_era5 = True` (hardcoded, cannot be False)

- **Code**: `data/real_data_loader.py`
  - Line 235: `"ONLY use real ERA5 data, never synthetic"`
  - Line 242-248: Uses `extract_frames_at_times()` (real ERA5 extraction)
  - Line 276-278: Returns `None` if ERA5 fails (NOT synthetic)
  - Line 280-281: Returns `None` if no ERA5 (NOT synthetic)
  - `_generate_synthetic_frames()` exists but is **NEVER CALLED**

### 2. Crop Size: 32x32 ✓
- **Location**: `data/generate_data_by_year.py` line 132
- **Code**: `image_size=(32, 32)  # 32x32 fits within all pre-cropped ERA5 files`

### 3. Format: .npy (like LT3P) ✓
- **Location**: `data/generate_data_by_year.py` lines 245-252
- **Code**: Uses `np.save()` for all components:
  - `{base_name}_past_frames.npy`
  - `{base_name}_future_frames.npy`
  - `{base_name}_track_past.npy`
  - `{base_name}_track_future.npy`
  - `{base_name}_intensity_past.npy`
  - `{base_name}_intensity_future.npy`
  - `{base_name}_pressure_past.npy`
  - `{base_name}_pressure_future.npy`
  - `{base_name}_meta.npz` (metadata)

### 4. xarray Merge Fixed ✓
- **Location**: `data/real_data_loader.py`
- **Fix**: Added `join='outer'` to all `xr.merge()` calls (lines 652, 735, 843, 849)
- **Status**: No more alignment errors

## Verification Results

- **Synthetic references in log**: **0** ✓
- **Process status**: Running (4.25 GB memory)
- **Code checks**: All passed

## Expected Output

When regeneration completes, check:

```powershell
# Check dataset info
Get-Content data/processed_temporal_split/dataset_info.json | ConvertFrom-Json | Select-Object meteorological_data, generation_date, total_samples

# Should show:
# - meteorological_data: "ERA5" (NOT "Synthetic")
# - generation_date: Today's date
# - total_samples: [number]
```

## Monitoring

To monitor the process:

```powershell
# Check process
Get-Process -Id 17460

# Check log for synthetic data
$log = Get-ChildItem "regeneration_log_20251120_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $log.FullName | Select-String -Pattern "synthetic|Synthetic"  # Should return nothing
```

## Guarantees

✅ **NO synthetic data will be generated**  
✅ **Only `extract_frames_at_times()` is called** (real ERA5)  
✅ **`_generate_synthetic_frames()` is NEVER called**  
✅ **Samples without ERA5 are skipped** (not replaced)  
✅ **If ERA5 extraction fails, sample is skipped** (not replaced)  
✅ **32x32 crop size**  
✅ **.npy format** (like LT3P)
