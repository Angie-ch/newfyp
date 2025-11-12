# IBTrACS Position Encoding Fix

## Problem Summary

The original IBTrACS encoding used a Gaussian blob centered at (0,0) multiplied by the normalized absolute lat/lon value. This approach had a critical flaw:

### Original (Broken) Approach
```python
# Created a Gaussian at center
position_marker = torch.exp(-dist_sq / (2 * sigma**2))

# Multiplied by scalar position
encoded[b, t, 0] = position_marker * lat_norm  # Same blob shape, different intensity
encoded[b, t, 1] = position_marker * lon_norm
```

**Problem**: Since ERA5 frames are already centered on the typhoon, every typhoon had the same Gaussian blob pattern at the center. The only difference was the intensity of the blob. This made it **impossible to decode absolute position** because:
- All typhoons looked identical spatially (blob at center)
- Position was encoded only in the blob's intensity, not its location
- The decoder couldn't distinguish between different lat/lon values

---

## Solution: Spatially-Varying Coordinate Fields

The fix creates **absolute coordinate grids** where each pixel encodes its own geographic coordinates.

### New (Fixed) Approach

#### Channel 0: Latitude Field (Vertical Gradient)
```python
for i in range(H):
    # Each row encodes its absolute latitude
    pixel_lat = lat_center + (i - H/2) * (lat_range / H)
    lat_norm = (pixel_lat - 22.5) / 12.5  # Normalize
    encoded[b, t, 0, i, :] = lat_norm  # All pixels in row i have same value
```

**Result**: A vertical gradient where:
- Top row (north): higher values (e.g., +0.8 for 32°N)
- Center row: 0.0 (e.g., 22.5°N)
- Bottom row (south): lower values (e.g., -0.8 for 13°N)

#### Channel 1: Longitude Field (Horizontal Gradient)
```python
for j in range(W):
    # Each column encodes its absolute longitude
    pixel_lon = lon_center + (j - W/2) * (lon_range / W)
    lon_norm = (pixel_lon - 140.0) / 20.0  # Normalize
    encoded[b, t, 1, :, j] = lon_norm  # All pixels in column j have same value
```

**Result**: A horizontal gradient where:
- Left column (west): lower values (e.g., -0.5 for 130°E)
- Center column: 0.0 (e.g., 140°E)
- Right column (east): higher values (e.g., +0.5 for 150°E)

#### Channels 2-3: Intensity & Pressure (Unchanged)
These remain as uniform fields (same value everywhere), which is fine because:
- They represent scalar properties of the typhoon
- No spatial variation needed
- Easily accessible by Conv layers at any location

---

## Decoding: Simple and Exact

With the new encoding, extracting position is trivial:

```python
# Extract from center pixel (typhoon location)
center_h, center_w = H // 2, W // 2

# Latitude from channel 0
lat_norm = encoded[:, :, 0, center_h, center_w]
lat = lat_norm * 12.5 + 22.5  # Denormalize

# Longitude from channel 1  
lon_norm = encoded[:, :, 1, center_h, center_w]
lon = lon_norm * 20.0 + 140.0  # Denormalize
```

**Result**: Perfect reconstruction with **zero error** (see test results).

---

## Visualization

The encoding creates these patterns:

```
Channel 0 (Latitude):        Channel 1 (Longitude):
┌─────────────────┐         ┌─────────────────┐
│ 0.8  0.8  0.8  │ North   │-0.5  0.0  +0.5 │
│ 0.4  0.4  0.4  │         │-0.5  0.0  +0.5 │
│ 0.0  0.0  0.0  │ Center  │-0.5  0.0  +0.5 │
│-0.4 -0.4 -0.4  │         │-0.5  0.0  +0.5 │
│-0.8 -0.8 -0.8  │ South   │-0.5  0.0  +0.5 │
└─────────────────┘         └─────────────────┘
   West → East                West → East

Channel 2 (Intensity):       Channel 3 (Pressure):
┌─────────────────┐         ┌─────────────────┐
│ 0.06 0.06 0.06 │         │-0.09-0.09-0.09 │
│ 0.06 0.06 0.06 │         │-0.09-0.09-0.09 │
│ 0.06 0.06 0.06 │         │-0.09-0.09-0.09 │
│ 0.06 0.06 0.06 │         │-0.09-0.09-0.09 │
│ 0.06 0.06 0.06 │         │-0.09-0.09-0.09 │
└─────────────────┘         └─────────────────┘
   Uniform field              Uniform field
```

---

## Benefits

### ✅ Position is Recoverable
- Center pixel directly encodes typhoon's absolute lat/lon
- Any pixel can tell you its geographic coordinates
- Perfect reconstruction (0.0° error in tests)

### ✅ Spatially Meaningful
- Each location "knows" where it is in the world
- Conv layers can learn position-dependent patterns
- Natural coordinate system for the model

### ✅ Joint Encoding Preserved
- IBTrACS and ERA5 concatenated as requested
- Single tensor input to the autoencoder
- Learned representations connect position with atmospheric patterns

### ✅ Efficient for CNNs
- Simple gradient patterns
- No complex structures to learn
- Compatible with any convolutional architecture

---

## Test Results

### 1. Basic Concatenation Test
```bash
python test_ibtracs_concat.py
```
**Result**: ✅ Channels increased from 48 → 52 (+4 IBTrACS)

### 2. Encoding/Decoding Accuracy Test
```bash
python test_position_encoding_decoding.py
```
**Result**: ✅ Mean Absolute Errors:
- Latitude: 0.0000° (< 0.5° threshold)
- Longitude: 0.0000° (< 0.5° threshold)
- Intensity: 0.0000 m/s (< 1.0 m/s threshold)
- Pressure: 0.0000 hPa (< 2.0 hPa threshold)

### 3. Visualization
```bash
python visualize_ibtracs_encoding.py
```
**Result**: ✅ Generated `ibtracs_encoding_visualization.png` showing the gradient structure

---

## Files Modified

1. **`data/utils/ibtracs_encoding.py`**
   - Updated `encode_track_as_channels()` to create coordinate grids
   - Updated `decode_track_from_channels()` to extract from center pixel
   - Updated docstrings to reflect new approach

2. **Test files created**:
   - `test_ibtracs_concat.py` - Tests dataset integration
   - `test_position_encoding_decoding.py` - Tests encode/decode accuracy
   - `visualize_ibtracs_encoding.py` - Creates visualization

---

## Usage in Training

The dataset now works correctly with `concat_ibtracs=True`:

```python
from data.datasets.typhoon_dataset import TyphoonDataset

dataset = TyphoonDataset(
    data_dir='data/processed',
    split='train',
    normalize=True,
    concat_ibtracs=True  # ← Enables the fixed encoding
)

sample = dataset[0]
print(sample['past_frames'].shape)  # (T, 52, H, W) - 48 ERA5 + 4 IBTrACS
```

The autoencoder can now:
1. Process atmospheric patterns (ERA5) and position info (IBTrACS) jointly
2. Learn how typhoon behavior depends on geographic location
3. Generate predictions with recoverable absolute positions

---

## Normalization Ranges

**Western Pacific Typhoon Assumptions**:
- Latitude: 10-35°N → normalized to [-1, +1] using center=22.5°, scale=12.5°
- Longitude: 120-160°E → normalized to [-1, +1] using center=140°, scale=20°
- Intensity: 17-70 m/s → normalized to [-1, +1] using center=43.5 m/s, scale=26.5 m/s
- Pressure: 900-1010 hPa → normalized to [-1, +1] using center=955 hPa, scale=55 hPa

These ranges can be adjusted if your data has different distributions.

---

## Summary

The fix transforms the position encoding from:
- ❌ **Static blob** (same pattern, different intensity) → Can't decode position
- ✅ **Coordinate grids** (spatial gradients) → Perfect position recovery

This enables the model to learn joint representations of atmospheric dynamics and geographic location while maintaining the ability to extract absolute typhoon positions from decoder outputs.

