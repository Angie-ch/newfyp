"""
Resample interpolated typhoon tracks from 1-hour to 6-hour intervals
to match LT3P paper requirements
"""
import pandas as pd
from pathlib import Path

def resample_to_6h_intervals(input_file: Path, output_file: Path):
    """Resample 1-hour interpolated tracks to 6-hour intervals"""
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original records: {len(df)}")
    print(f"Original storms: {df['typhoon_id'].nunique()}")
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.drop('time', axis=1)
    elif 'ISO_TIME' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ISO_TIME'])
    else:
        # Assume first column or check for datetime-like column
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("No timestamp column found (looked for: timestamp, time, ISO_TIME)")
    
    # Resample each storm to 6-hour intervals
    resampled_dfs = []
    
    for storm_id in df['typhoon_id'].unique():
        storm_df = df[df['typhoon_id'] == storm_id].copy()
        storm_df = storm_df.set_index('timestamp').sort_index()
        
        # Resample to 6-hour intervals (keep first value in each 6-hour window)
        storm_resampled = storm_df.resample('6H').first()
        
        # Drop rows with NaN (incomplete 6-hour windows)
        storm_resampled = storm_resampled.dropna(subset=['lat', 'lon'])
        
        # Reset index to make timestamp a column again
        storm_resampled = storm_resampled.reset_index()
        
        resampled_dfs.append(storm_resampled)
    
    # Combine all storms
    result_df = pd.concat(resampled_dfs, ignore_index=True)
    
    print(f"\nResampled records: {len(result_df)}")
    print(f"Resampled storms: {result_df['typhoon_id'].nunique()}")
    print(f"Avg records per storm: {len(result_df) / result_df['typhoon_id'].nunique():.1f}")
    
    # Save
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    return result_df


if __name__ == "__main__":
    data_dir = Path("data/raw")
    input_file = data_dir / "interpolated_typhoon_tracks_2018_2021.csv"
    output_file = data_dir / "interpolated_typhoon_tracks_2018_2021_6h.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        exit(1)
    
    resample_to_6h_intervals(input_file, output_file)
    
    print("\n" + "="*80)
    print("6-HOUR RESAMPLING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Use the resampled file in generate_data_by_year.py")
    print("2. Update ERA5 loading to match 6-hour intervals")
    print("3. Run regeneration with --use-6h-data flag")

