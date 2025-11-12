"""
Prepare interpolated typhoon tracks from processed tracks dataset.

This script:
1. Extracts typhoon tracks from processed_typhoon_tracks_1950_2017.csv
2. Filters for Western Pacific typhoons in specified years
3. Interpolates 3-hourly data to 1-hourly intervals to match ERA5
4. Saves the interpolated tracks in IBTrACS-compatible format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import interp1d


def parse_typhoon_time(time_code: int, year: int) -> datetime:
    """
    Parse the time code from processed tracks.
    
    Format: DDDHHSS where:
    - DDD: day of year (1-366)
    - HH: hour (00-23)
    - SS: unknown (usually 00)
    
    Example: 60500 = day 6, hour 05, 00
    """
    time_str = str(time_code).zfill(6)
    day_of_year = int(time_str[:3])
    hour = int(time_str[3:5])
    
    # Create datetime from year and day of year
    base_date = datetime(year, 1, 1)
    target_date = base_date + timedelta(days=day_of_year - 1, hours=hour)
    
    return target_date


def interpolate_track(storm_df: pd.DataFrame, freq='1H') -> pd.DataFrame:
    """
    Interpolate typhoon track to specified frequency (e.g., 1-hour intervals).
    
    Args:
        storm_df: DataFrame with columns [time_dt, lat, lon, wind, pressure]
        freq: Pandas frequency string (default '1H' for 1-hour)
    
    Returns:
        DataFrame with interpolated values at specified frequency
    """
    # Sort by time
    storm_df = storm_df.sort_values('time_dt').reset_index(drop=True)
    
    # Remove duplicates at same timestamp
    storm_df = storm_df.drop_duplicates(subset=['time_dt'], keep='first')
    
    if len(storm_df) < 2:
        return storm_df  # Can't interpolate with less than 2 points
    
    # Create time range at specified frequency
    start_time = storm_df['time_dt'].min()
    end_time = storm_df['time_dt'].max()
    new_times = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Convert times to numeric (seconds since start) for interpolation
    time_numeric = (storm_df['time_dt'] - start_time).dt.total_seconds().values
    new_time_numeric = (new_times - start_time).total_seconds().values
    
    # Interpolate each variable
    interpolated_data = {'time_dt': new_times}
    
    # Latitude and longitude - linear interpolation
    lat_interp = interp1d(time_numeric, storm_df['lat'].values, 
                         kind='linear', fill_value='extrapolate')
    lon_interp = interp1d(time_numeric, storm_df['lon'].values, 
                         kind='linear', fill_value='extrapolate')
    
    interpolated_data['lat'] = lat_interp(new_time_numeric)
    interpolated_data['lon'] = lon_interp(new_time_numeric)
    
    # Wind speed - linear interpolation (could use quadratic for smoothness)
    wind_interp = interp1d(time_numeric, storm_df['wind'].values, 
                          kind='linear', fill_value='extrapolate')
    interpolated_data['wind'] = wind_interp(new_time_numeric)
    
    # Pressure - linear interpolation
    pressure_interp = interp1d(time_numeric, storm_df['pressure'].values, 
                              kind='linear', fill_value='extrapolate')
    interpolated_data['pressure'] = pressure_interp(new_time_numeric)
    
    # Ensure non-negative wind speeds
    interpolated_data['wind'] = np.maximum(interpolated_data['wind'], 0)
    
    # Ensure reasonable pressure values (900-1020 hPa)
    interpolated_data['pressure'] = np.clip(interpolated_data['pressure'], 900, 1020)
    
    result_df = pd.DataFrame(interpolated_data)
    
    # Copy over storm metadata
    for col in ['typhoon_id', 'typhoon_name', 'year']:
        if col in storm_df.columns:
            result_df[col] = storm_df[col].iloc[0]
    
    return result_df


def main():
    """Main processing function."""
    
    # Configuration
    START_YEAR = 2018
    END_YEAR = 2021
    MIN_WIND_SPEED = 33.0  # m/s (~64 knots, typhoon threshold)
    MIN_DURATION_HOURS = 48
    INTERPOLATION_FREQ = '1H'  # 1-hour intervals
    
    INPUT_FILE = Path("data/raw/processed_typhoon_tracks_1950_2017.csv")
    OUTPUT_DIR = Path("data/raw")
    OUTPUT_FILE = OUTPUT_DIR / "interpolated_typhoon_tracks_2018_2021.csv"
    
    print("="*80)
    print("PREPARING INTERPOLATED TYPHOON TRACKS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Year range: {START_YEAR}-{END_YEAR}")
    print(f"  Min wind speed: {MIN_WIND_SPEED} m/s")
    print(f"  Min duration: {MIN_DURATION_HOURS} hours")
    print(f"  Interpolation frequency: {INTERPOLATION_FREQ}")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output file: {OUTPUT_FILE}")
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df)} records")
    print(f"  Unique storms: {df['typhoon_id'].nunique()}")
    
    # Filter by year
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    print(f"\n✓ Filtered to {len(df)} records ({START_YEAR}-{END_YEAR})")
    print(f"  Unique storms: {df['typhoon_id'].nunique()}")
    
    # Parse timestamps
    print("\nParsing timestamps...")
    df['time_dt'] = df.apply(lambda row: parse_typhoon_time(row['time'], row['year']), axis=1)
    print("✓ Timestamps parsed")
    
    # Filter strong typhoons
    print(f"\nFiltering strong typhoons (wind >= {MIN_WIND_SPEED} m/s)...")
    storm_max_winds = df.groupby('typhoon_id')['wind'].max()
    strong_storms = storm_max_winds[storm_max_winds >= MIN_WIND_SPEED].index.tolist()
    
    # Filter by duration
    storm_durations = df.groupby('typhoon_id')['time_dt'].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 3600
    )
    long_storms = storm_durations[storm_durations >= MIN_DURATION_HOURS].index.tolist()
    
    # Get intersection
    selected_storms = list(set(strong_storms) & set(long_storms))
    df_filtered = df[df['typhoon_id'].isin(selected_storms)]
    
    print(f"✓ Found {len(selected_storms)} strong typhoons:")
    print(f"  Max wind >= {MIN_WIND_SPEED} m/s")
    print(f"  Duration >= {MIN_DURATION_HOURS} hours")
    print(f"  Total records: {len(df_filtered)}")
    
    # Show storm summary
    print(f"\nStorm summary:")
    storm_summary = df_filtered.groupby(['typhoon_id', 'typhoon_name', 'year']).agg({
        'wind': 'max',
        'pressure': 'min',
        'time_dt': ['min', 'max', 'count']
    }).reset_index()
    storm_summary.columns = ['typhoon_id', 'typhoon_name', 'year', 'max_wind', 
                             'min_pressure', 'start_time', 'end_time', 'num_records']
    storm_summary['duration_hours'] = (
        (storm_summary['end_time'] - storm_summary['start_time']).dt.total_seconds() / 3600
    )
    
    print(storm_summary[['typhoon_id', 'typhoon_name', 'year', 'max_wind', 
                         'min_pressure', 'num_records', 'duration_hours']].to_string(index=False))
    
    # Interpolate each storm
    print(f"\nInterpolating storms to {INTERPOLATION_FREQ} intervals...")
    interpolated_storms = []
    
    for storm_id in selected_storms:
        storm_df = df_filtered[df_filtered['typhoon_id'] == storm_id].copy()
        
        try:
            interpolated = interpolate_track(storm_df, freq=INTERPOLATION_FREQ)
            interpolated_storms.append(interpolated)
            
            original_count = len(storm_df)
            interpolated_count = len(interpolated)
            print(f"  {storm_id}: {original_count} → {interpolated_count} records")
            
        except Exception as e:
            print(f"  ✗ Error interpolating {storm_id}: {e}")
            continue
    
    # Combine all interpolated storms
    if interpolated_storms:
        df_interpolated = pd.concat(interpolated_storms, ignore_index=True)
        
        print(f"\n✓ Interpolation complete!")
        print(f"  Total records: {len(df_interpolated)}")
        print(f"  Storms: {len(interpolated_storms)}")
        
        # Add ISO_TIME column for compatibility
        df_interpolated['ISO_TIME'] = df_interpolated['time_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns
        output_columns = ['typhoon_id', 'typhoon_name', 'ISO_TIME', 'year', 
                         'lat', 'lon', 'wind', 'pressure']
        df_output = df_interpolated[output_columns]
        
        # Sort by storm and time
        df_output = df_output.sort_values(['typhoon_id', 'ISO_TIME'])
        
        # Save to file
        print(f"\nSaving to {OUTPUT_FILE}...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Saved {len(df_output)} records to {OUTPUT_FILE}")
        
        # Show sample
        print("\nSample of interpolated data:")
        sample_storm = df_output['typhoon_id'].iloc[0]
        print(df_output[df_output['typhoon_id'] == sample_storm].head(10).to_string(index=False))
        
        print("\n" + "="*80)
        print("✓ INTERPOLATION COMPLETE!")
        print("="*80)
        print(f"\nOutput file: {OUTPUT_FILE}")
        print(f"Total records: {len(df_output)}")
        print(f"Storms: {len(interpolated_storms)}")
        print(f"Year range: {df_output['year'].min()}-{df_output['year'].max()}")
        print(f"Temporal resolution: {INTERPOLATION_FREQ} (hourly)")
        print(f"\nThis dataset is now compatible with ERA5 1-hourly data!")
        
    else:
        print("\n✗ No storms were successfully interpolated!")


if __name__ == "__main__":
    main()

