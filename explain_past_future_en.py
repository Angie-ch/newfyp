"""
How Past and Future Frames are Separated

Example: A typhoon with 42 timesteps (6-hour intervals)
Config:
  - PAST_TIMESTEPS = 2 (2 steps = 12 hours history)
  - FUTURE_TIMESTEPS = 4 (4 steps = 24 hours forecast)
  - STRIDE = 1 (slide 1 step at a time)
  - SKIP_EARLY_TIMESTEPS = 4 (skip first 4 to ensure ERA5 history exists)
"""

# Simulate typhoon time series
total_timesteps = 42
PAST_TIMESTEPS = 2
FUTURE_TIMESTEPS = 4
STRIDE = 1
SKIP_EARLY_TIMESTEPS = 4

print("="*80)
print("HOW PAST AND FUTURE FRAMES ARE SEPARATED")
print("="*80)

print(f"\nTyphoon Configuration:")
print(f"  Total timesteps: {total_timesteps}")
print(f"  Past timesteps: {PAST_TIMESTEPS} (12-hour history)")
print(f"  Future timesteps: {FUTURE_TIMESTEPS} (24-hour forecast)")
print(f"  Sliding window stride: {STRIDE}")
print(f"  Skip first: {SKIP_EARLY_TIMESTEPS} steps")

print(f"\nGenerated Samples:")
print("-"*80)

sample_count = 0
for start_idx in range(SKIP_EARLY_TIMESTEPS, 
                       total_timesteps - PAST_TIMESTEPS - FUTURE_TIMESTEPS + 1, 
                       STRIDE):
    
    # Calculate index ranges
    past_start = start_idx
    past_end = start_idx + PAST_TIMESTEPS
    future_start = past_end
    future_end = future_start + FUTURE_TIMESTEPS
    
    sample_count += 1
    
    if sample_count <= 5 or sample_count > 30:  # Show first 5 and last few
        print(f"\nSample #{sample_count}:")
        print(f"  start_idx = {start_idx}")
        print(f"  Past indices: [{past_start}:{past_end}]  (timesteps {past_start}, {past_start+1})")
        print(f"  Future indices: [{future_start}:{future_end}]  (timesteps {future_start}, {future_start+1}, {future_start+2}, {future_start+3})")
        print(f"  Total span: timestep {past_start} to {future_end-1} ({future_end-past_start} steps = {(future_end-past_start)*6} hours)")
    elif sample_count == 6:
        print(f"\n  ... (middle samples omitted) ...")

print(f"\nTotal samples generated: {sample_count}")

print("\n" + "="*80)
print("CODE IMPLEMENTATION")
print("="*80)

print("""
In data/real_data_loader.py, create_training_sample() function:

STEP 1: Define time indices
----------------------------
past_start = start_idx
past_end = start_idx + past_timesteps
future_start = past_end
future_end = future_start + future_timesteps

STEP 2: Slice typhoon track data
---------------------------------
# Past trajectory and intensity
past_lats = storm_data['lats'][past_start:past_end]
past_lons = storm_data['lons'][past_start:past_end]
past_times = storm_data['times'][past_start:past_end]

# Future trajectory and intensity
future_lats = storm_data['lats'][future_start:future_end]
future_lons = storm_data['lons'][future_start:future_end]
future_times = storm_data['times'][future_start:future_end]

STEP 3: Extract ERA5 meteorological fields
-------------------------------------------
# Extract past ERA5 frames
past_frames = era5_loader.extract_frames_at_times(
    center_lons=past_lons,
    center_lats=past_lats,
    times=past_times,
    crop_size=64,
    load_per_timestep=True  # Load individual files per timestep
)
# Shape: (2, 24, 64, 64)
#        ^  ^   ^   ^
#        |  |   |   +-- Width 64 pixels
#        |  |   +------ Height 64 pixels
#        |  +---------- 24 meteorological channels
#        +------------- 2 past timesteps

# Extract future ERA5 frames
future_frames = era5_loader.extract_frames_at_times(
    center_lons=future_lons,
    center_lats=future_lats,
    times=future_times,
    crop_size=64,
    load_per_timestep=True
)
# Shape: (4, 24, 64, 64)
#        ^  ^   ^   ^
#        |  |   |   +-- Width 64 pixels
#        |  |   +------ Height 64 pixels
#        |  +---------- 24 meteorological channels
#        +------------- 4 future timesteps

STEP 4: Save to .npz file
--------------------------
np.savez(
    output_file,
    past_frames=past_frames,        # (2, 24, 64, 64)
    future_frames=future_frames,    # (4, 24, 64, 64)
    track_past=track_past,          # (2, 2) - [lat, lon]
    track_future=track_future,      # (4, 2) - [lat, lon]
    intensity_past=intensity_past,  # (2,) - wind speed
    intensity_future=intensity_future  # (4,) - wind speed
)
""")

print("\n" + "="*80)
print("KEY POINTS")
print("="*80)
print("""
1. TIME CONTINUITY:
   - Past frames end time = Future frames start time
   - No time gap between past and future
   - Example: past[0,1], future[2,3,4,5]

2. SLIDING WINDOW:
   - STRIDE=1: Move forward 1 timestep each time
   - Creates overlapping samples to increase training data
   - Example: Sample1[0:6], Sample2[1:7], Sample3[2:8]...

3. SPATIAL ALIGNMENT:
   - Each timestep's ERA5 frame is cropped around typhoon center
   - 64x64 pixels covers ~16deg x 16deg region
   - Follows typhoon movement

4. USAGE:
   - Model Input: past_frames (historical meteorological fields)
   - Model Output: future_frames (prediction target)
   - Or for trajectory prediction: track_past -> track_future

5. PER-TIMESTEP LOADING:
   - Each timestep loads its own ERA5 file for that date
   - Avoids NaN from merging files with different spatial coverage
   - Ensures 100% real ERA5 data
""")

print("="*80)
print("\nVISUAL EXAMPLE:")
print("="*80)
print("""
Typhoon timeline (6-hour intervals):
0---1---2---3---4---5---6---7---8---9---10--11--12 ... 42
    |       |               |
    skip    Sample #1       Sample #2
    first   past: [4,5]     past: [5,6]
    4 steps future:[6,7,8,9] future:[7,8,9,10]

Result:
- Sample #1: Uses timesteps 4,5 as input to predict 6,7,8,9
- Sample #2: Uses timesteps 5,6 as input to predict 7,8,9,10
- Each sample is a complete input-output pair for training
""")

print("="*80)

