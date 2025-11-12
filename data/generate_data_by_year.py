"""
Generate typhoon dataset with proper temporal split by year

This script creates a dataset split by year to avoid data leakage:
- Training: 2018-2019
- Validation: 2020
- Test: 2021

This ensures that the same typhoon does not appear in both training and test sets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import pickle
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from real_data_loader import IBTrACSLoader, ERA5Loader


def extract_year_from_storm_id(storm_id: str) -> int:
    """
    Extract year from IBTrACS storm ID
    
    Storm ID format: YYYY-BASIN-NUMBER (e.g., "2021WP01")
    or: YYYYBASINNUMBER
    """
    # Try different formats
    if '-' in storm_id:
        year_str = storm_id.split('-')[0]
    else:
        # Format like "2021WP01"
        year_str = storm_id[:4]
    
    try:
        return int(year_str)
    except ValueError:
        print(f"Warning: Could not extract year from storm ID: {storm_id}")
        return None


def generate_samples_by_storm(
    ibtracs_loader: IBTrACSLoader,
    df: pd.DataFrame,
    storm_ids: List[str],
    past_timesteps: int = 8,
    future_timesteps: int = 12,
    stride: int = None,
    era5_loader: Optional[ERA5Loader] = None,
    era5_datasets: Optional[Dict] = None
) -> tuple:
    """
    Generate training samples organized by year using systematic sliding windows
    
    Args:
        ibtracs_loader: IBTrACS data loader
        df: IBTrACS dataframe
        storm_ids: List of storm IDs to process
        past_timesteps: Number of past timesteps
        future_timesteps: Number of future timesteps
        stride: Stride for sliding window (None = non-overlapping)
        era5_loader: ERA5 loader instance (optional)
        era5_datasets: Pre-loaded ERA5 datasets by storm_id (optional)
    
    Returns:
        Tuple of (samples_by_year dict, samples_by_storm dict)
    """
    samples_by_year = defaultdict(list)
    samples_by_storm = {}  # Track samples per storm for organization
    
    print(f"\nGenerating systematic samples for {len(storm_ids)} storms...")
    print(f"  Past timesteps: {past_timesteps}")
    print(f"  Future timesteps: {future_timesteps}")
    print(f"  Stride: {stride if stride else 'non-overlapping (stride=' + str(past_timesteps) + ')'}")
    
    storms_processed = 0
    storms_skipped = 0
    
    for storm_id in storm_ids:
        # Extract year
        year = extract_year_from_storm_id(storm_id)
        if year is None:
            storms_skipped += 1
            continue
        
        # Get storm data
        storm_data = ibtracs_loader.get_storm_data(df, storm_id)
        
        if len(storm_data['times']) < past_timesteps + future_timesteps:
            storms_skipped += 1
            continue
        
        # Get ERA5 data - REQUIRED, no synthetic data allowed
        if not era5_datasets or storm_id not in era5_datasets:
            # Skip this storm if no ERA5 data available
            storms_skipped += 1
            continue
        
        era5_dataset = era5_datasets[storm_id]
        use_era5 = True
        
        # Generate ALL systematic samples from this storm using sliding windows
        storm_samples = []
        n_total = len(storm_data['times'])
        
        # Determine stride
        actual_stride = stride if stride is not None else past_timesteps
        
        # Generate sliding window samples
        for start_idx in range(0, n_total - past_timesteps - future_timesteps + 1, actual_stride):
            # Create a sample starting at this index by slicing the storm data
            sliced_storm_data = {
                'storm_id': storm_data['storm_id'],
                'name': storm_data.get('name', 'UNKNOWN'),
                'times': storm_data['times'][start_idx:start_idx + past_timesteps + future_timesteps],
                'lats': storm_data['lats'][start_idx:start_idx + past_timesteps + future_timesteps],
                'lons': storm_data['lons'][start_idx:start_idx + past_timesteps + future_timesteps],
                'winds': storm_data['winds'][start_idx:start_idx + past_timesteps + future_timesteps],
                'pressures': storm_data['pressures'][start_idx:start_idx + past_timesteps + future_timesteps]
            }
            
            # Create sample with the sliced data (this will use start_idx=0 internally)
            sample = ibtracs_loader.create_training_sample(
                sliced_storm_data,
                past_timesteps=past_timesteps,
                future_timesteps=future_timesteps,
                era5_dataset=era5_dataset,
                era5_loader=era5_loader,
                use_era5=use_era5
            )
            
            if sample is not None:
                # Add metadata for this specific window
                sample['window_index'] = len(storm_samples)
                sample['start_idx'] = start_idx
                storm_samples.append(sample)
        
        # Add metadata and organize by year
        for sample in storm_samples:
            sample['year'] = year
            samples_by_year[year].append(sample)
        
        samples_by_storm[storm_id] = storm_samples
        
        storms_processed += 1
        if storms_processed % 10 == 0:
            total_samples = sum(len(v) for v in samples_by_year.values())
            print(f"  Processed {storms_processed}/{len(storm_ids)} storms, {total_samples} samples generated...")
    
    total_samples = sum(len(v) for v in samples_by_year.values())
    print(f"\n✓ Generated {total_samples} samples from {storms_processed} storms")
    print(f"  Skipped {storms_skipped} storms (insufficient data)")
    
    # Print distribution by year
    print("\nSamples by year:")
    for year in sorted(samples_by_year.keys()):
        n_samples = len(samples_by_year[year])
        unique_storms = len(set(s['storm_id'] for s in samples_by_year[year]))
        print(f"  {year}: {n_samples} samples from {unique_storms} storms")
    
    return dict(samples_by_year), samples_by_storm


def save_samples_by_split(
    samples_by_year: Dict[int, List[Dict]],
    samples_by_storm: Dict[str, List[Dict]],
    output_dir: Path,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int]
):
    """
    Save samples to separate directories by split, organized by typhoon ID
    
    Args:
        samples_by_year: Dictionary mapping year -> samples
        samples_by_storm: Dictionary mapping storm_id -> samples  
        output_dir: Base output directory
        train_years: Years for training set
        val_years: Years for validation set
        test_years: Years for test set
    """
    # Create output directories
    splits = {
        'train': train_years,
        'val': val_years,
        'test': test_years
    }
    
    print(f"\nSaving samples to {output_dir}...")
    print(f"  Train years: {train_years}")
    print(f"  Val years: {val_years}")
    print(f"  Test years: {test_years}")
    
    # Group samples by storm and year for organization
    storm_to_split = {}
    for split_name, years in splits.items():
        for year in years:
            if year in samples_by_year:
                for sample in samples_by_year[year]:
                    storm_id = sample['storm_id']
                    storm_to_split[storm_id] = split_name
    
    for split_name, years in splits.items():
        split_dir = output_dir / split_name / 'cases'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        storm_count = 0
        
        for year in years:
            if year not in samples_by_year:
                print(f"  Warning: No samples found for year {year}")
                continue
            
            year_samples = samples_by_year[year]
            
            # Group by storm for this year
            storms_in_year = {}
            for sample in year_samples:
                storm_id = sample['storm_id']
                if storm_id not in storms_in_year:
                    storms_in_year[storm_id] = []
                storms_in_year[storm_id].append(sample)
            
            # Save samples organized by typhoon
            for storm_id, storm_samples in storms_in_year.items():
                storm_count += 1
                storm_id_clean = storm_id.replace(' ', '_').replace('/', '_')
                
                for idx, sample in enumerate(storm_samples):
                    # Filename: YEAR_STORMID_WINDOW_INDEX.npz
                    window_idx = sample.get('window_index', idx)
                    filename = f"{year}_{storm_id_clean}_w{window_idx:02d}.npz"
                    
                    filepath = split_dir / filename
                    
                    # Save sample
                    np.savez_compressed(
                        filepath,
                        past_frames=sample['past_frames'],
                        future_frames=sample['future_frames'],
                        track_past=sample['past_track'],
                        track_future=sample['future_track'],
                        intensity_past=sample['past_intensity'],
                        intensity_future=sample['future_intensity'],
                        pressure_past=sample['past_pressure'],
                        pressure_future=sample['future_pressure'],
                        case_id=f"{year}_{storm_id_clean}_w{window_idx:02d}",
                        storm_id=sample['storm_id'],
                        storm_name=sample['storm_name'],
                        year=year,
                        window_index=window_idx,
                        start_idx=sample.get('start_idx', 0)
                    )
                    
                    sample_count += 1
        
        print(f"  ✓ Saved {sample_count} samples from {storm_count} storms to {split_name}/ directory")
    
    # Save metadata
    metadata = {
        'train_years': train_years,
        'val_years': val_years,
        'test_years': test_years,
        'n_train': sum(len(samples_by_year.get(y, [])) for y in train_years),
        'n_val': sum(len(samples_by_year.get(y, [])) for y in val_years),
        'n_test': sum(len(samples_by_year.get(y, [])) for y in test_years),
        'past_timesteps': 8,
        'future_timesteps': 12
    }
    
    with open(output_dir / 'dataset_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Saved metadata to {output_dir / 'dataset_metadata.pkl'}")
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Training samples:   {metadata['n_train']} (years {train_years})")
    print(f"Validation samples: {metadata['n_val']} (years {val_years})")
    print(f"Test samples:       {metadata['n_test']} (years {test_years})")
    print(f"\nTotal: {metadata['n_train'] + metadata['n_val'] + metadata['n_test']} samples")


def load_era5_for_storms(storm_ids: List[str], df: pd.DataFrame, ibtracs_loader: IBTrACSLoader):
    """
    Load ERA5 data for storms from daily files organized by year
    
    Args:
        storm_ids: List of storm IDs
        df: IBTrACS dataframe
        ibtracs_loader: IBTrACS loader instance
    
    Returns:
        Tuple of (era5_loader, era5_datasets dict)
    """
    print("\n" + "="*80)
    print("LOADING ERA5 REANALYSIS DATA FROM DAILY FILES")
    print("="*80)
    
    era5_loader = ERA5Loader()
    era5_datasets = {}
    
    # Check if ERA5 directories exist
    era5_years = []
    for year in [2018, 2019, 2020, 2021]:
        year_dir = era5_loader.data_dir / f"ERA5_{year}_26data"
        if year_dir.exists():
            era5_years.append(year)
    
    if not era5_years:
        print(f"\n⚠️  WARNING: No ERA5 data directories found in {era5_loader.data_dir}")
        print("  Expected directories like: ERA5_2018_26data, ERA5_2019_26data, etc.")
        print("  Continuing without ERA5 data - will skip storms without ERA5 data.")
        return era5_loader, {}
    
    print(f"Found ERA5 data for years: {era5_years}")
    print(f"Loading ERA5 data for {len(storm_ids)} storms...")
    
    loaded_count = 0
    for idx, storm_id in enumerate(storm_ids):
        try:
            storm_data = ibtracs_loader.get_storm_data(df, storm_id)
            
            # Get storm time range with some buffer
            start_time = pd.to_datetime(storm_data['times'][0]) - pd.Timedelta(hours=6)
            end_time = pd.to_datetime(storm_data['times'][-1]) + pd.Timedelta(hours=6)
            
            # Get spatial extent with buffer
            lats = storm_data['lats']
            lons = storm_data['lons']
            lat_range = (float(np.min(lats) - 10), float(np.max(lats) + 10))
            lon_range = (float(np.min(lons) - 10), float(np.max(lons) + 10))
            
            # Load ERA5 data from daily files
            era5_ds = era5_loader.load_era5_from_daily_files(
                start_time=start_time,
                end_time=end_time,
                lat_range=lat_range,
                lon_range=lon_range
            )
            
            if era5_ds is not None:
                era5_datasets[storm_id] = era5_ds
                loaded_count += 1
                if loaded_count % 10 == 0:
                    print(f"  Loaded ERA5 for {loaded_count}/{len(storm_ids)} storms...")
        
        except Exception as e:
            print(f"  ✗ Failed to load ERA5 for {storm_id}: {e}")
            continue
    
    print(f"\n✓ Successfully loaded ERA5 data for {loaded_count}/{len(storm_ids)} storms")
    
    if loaded_count == 0:
        print("\n⚠️  WARNING: No ERA5 data could be loaded for any storms.")
        print("  Continuing without ERA5 data - will skip storms without ERA5 data.")
        print("  Training will proceed with available data only.")
    
    return era5_loader, era5_datasets


def check_all_era5_data_exists(
    ibtracs_loader: IBTrACSLoader,
    df: pd.DataFrame,
    storm_ids: List[str],
    era5_loader: Optional[ERA5Loader] = None,
    verbose: bool = True) -> Dict:
    """
    Check if all required ERA5 data files exist for the given storms
    
    Args:
        ibtracs_loader: IBTrACSLoader instance
        df: IBTrACS DataFrame
        storm_ids: List of storm IDs to check
        era5_loader: ERA5Loader instance (creates new one if None)
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with check results:
        {
            'all_exist': bool,
            'missing_directories': List[str],
            'storms_checked': int,
            'storms_with_all_data': int,
            'storms_with_missing_data': int,
            'missing_files_by_storm': Dict[str, List[str]],
            'total_missing_files': int
        }
    """
    if era5_loader is None:
        era5_loader = ERA5Loader()
    
    results = {
        'all_exist': True,
        'missing_directories': [],
        'storms_checked': 0,
        'storms_with_all_data': 0,
        'storms_with_missing_data': 0,
        'missing_files_by_storm': {},
        'total_missing_files': 0
    }
    
    if verbose:
        print("\n" + "="*80)
        print("CHECKING ERA5 DATA AVAILABILITY")
        print("="*80)
    
    # Check if ERA5 year directories exist
    required_years = [2018, 2019, 2020, 2021]
    existing_years = []
    
    for year in required_years:
        year_dir = era5_loader.data_dir / f"ERA5_{year}_26data"
        if year_dir.exists():
            existing_years.append(year)
        else:
            results['missing_directories'].append(f"ERA5_{year}_26data")
            results['all_exist'] = False
    
    if verbose:
        if existing_years:
            print(f"\n✓ Found ERA5 directories for years: {existing_years}")
        if results['missing_directories']:
            print(f"✗ Missing ERA5 directories: {results['missing_directories']}")
    
    if not existing_years:
        if verbose:
            print("\n❌ No ERA5 data directories found. Cannot check individual files.")
        return results
    
    # Check files for each storm
    if verbose:
        print(f"\nChecking ERA5 files for {len(storm_ids)} storms...")
    
    for idx, storm_id in enumerate(storm_ids):
        try:
            storm_data = ibtracs_loader.get_storm_data(df, storm_id)
            results['storms_checked'] += 1
            
            # Get storm time range with buffer
            start_time = pd.to_datetime(storm_data['times'][0]) - pd.Timedelta(hours=6)
            end_time = pd.to_datetime(storm_data['times'][-1]) + pd.Timedelta(hours=6)
            
            # Generate list of dates to check
            dates = pd.date_range(start=start_time.date(), end=end_time.date(), freq='D')
            
            missing_files = []
            
            for date in dates:
                year = date.year
                date_str = date.strftime('%Y%m%d')
                
                # Check if file exists (handle both single files and split files)
                year_dir = era5_loader.data_dir / f"ERA5_{year}_26data"
                file_path = year_dir / f"era5_pl_{date_str}.nc"
                file_path_part1 = year_dir / f"era5_pl_{date_str}_part1.nc"
                file_path_part2 = year_dir / f"era5_pl_{date_str}_part2.nc"
                
                # File exists if: single file exists OR (part1 exists AND part2 exists)
                file_exists = (file_path.exists() or 
                              (file_path_part1.exists() and file_path_part2.exists()))
                
                if not file_exists:
                    missing_files.append(str(file_path.relative_to(era5_loader.data_dir)))
            
            if missing_files:
                results['storms_with_missing_data'] += 1
                results['missing_files_by_storm'][storm_id] = missing_files
                results['total_missing_files'] += len(missing_files)
                results['all_exist'] = False
                
                if verbose:
                    print(f"  ✗ {storm_id}: Missing {len(missing_files)} files")
                    if len(missing_files) <= 5:
                        for f in missing_files:
                            print(f"      - {f}")
            else:
                results['storms_with_all_data'] += 1
            
            if verbose and (idx + 1) % 50 == 0:
                print(f"  Checked {idx + 1}/{len(storm_ids)} storms...")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Error checking {storm_id}: {e}")
            continue
    
    # Print summary
    if verbose:
        print("\n" + "="*80)
        print("ERA5 DATA CHECK SUMMARY")
        print("="*80)
        print(f"Storms checked: {results['storms_checked']}")
        print(f"Storms with all data: {results['storms_with_all_data']}")
        print(f"Storms with missing data: {results['storms_with_missing_data']}")
        print(f"Total missing files: {results['total_missing_files']}")
        
        if results['missing_directories']:
            print(f"\nMissing directories: {len(results['missing_directories'])}")
            for dir_name in results['missing_directories']:
                print(f"  - {dir_name}")
        
        if results['all_exist']:
            print("\n✓ All required ERA5 data files exist!")
        else:
            print("\n✗ Some ERA5 data files are missing.")
            if results['storms_with_missing_data'] > 0:
                print(f"\nStorms with missing data (organized by typhoon ID):")
                shown = 0
                for storm_id, files in results['missing_files_by_storm'].items():
                    if shown >= 20:
                        print(f"  ... and {len(results['missing_files_by_storm']) - shown} more storms")
                        break
                    print(f"\n  {storm_id}: {len(files)} missing file(s)")
                    for f in files:
                        print(f"      - {f}")
                    shown += 1
    
    return results


def main():
    """Main function to generate dataset with temporal split"""
    
    print("="*80)
    print("GENERATING TYPHOON DATASET WITH TEMPORAL SPLIT BY YEAR")
    print("="*80)
    
    # Configuration
    START_YEAR = 2018
    END_YEAR = 2021
    PAST_TIMESTEPS = 8     # 8 past timesteps (8 hours at 1-hour intervals)
    FUTURE_TIMESTEPS = 12  # 12 future timesteps (12 hours at 1-hour intervals)
    STRIDE = None          # None = non-overlapping windows (stride = PAST_TIMESTEPS)
    
    # Split configuration (use interpolated tracks: 2018-2021)
    TRAIN_YEARS = [2018, 2019]  # Training: 2018-2019
    VAL_YEARS = [2020]          # Validation: 2020
    TEST_YEARS = [2021]         # Test: 2021
    
    # Use absolute paths
    SCRIPT_DIR = Path(__file__).parent
    OUTPUT_DIR = SCRIPT_DIR / "processed_temporal_split"
    
    print(f"\nConfiguration:")
    print(f"  Year range: {START_YEAR}-{END_YEAR}")
    print(f"  Past timesteps: {PAST_TIMESTEPS}")
    print(f"  Future timesteps: {FUTURE_TIMESTEPS}")
    print(f"  Window stride: {'non-overlapping' if STRIDE is None else STRIDE}")
    print(f"  Train years: {TRAIN_YEARS}")
    print(f"  Val years: {VAL_YEARS}")
    print(f"  Test years: {TEST_YEARS}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Initialize loader with interpolated tracks
    ibtracs_loader = IBTrACSLoader(
        data_dir=str(SCRIPT_DIR / "raw")
    )
    
    # Load interpolated IBTrACS data
    interpolated_file = SCRIPT_DIR / "raw" / "interpolated_typhoon_tracks_2018_2021.csv"
    if interpolated_file.exists():
        print(f"Loading interpolated tracks from {interpolated_file}")
        df = pd.read_csv(interpolated_file, low_memory=False)
        print(f"✓ Loaded {len(df)} interpolated records")
        
        # Map column names to match IBTrACS format
        if 'typhoon_id' in df.columns:
            df['SID'] = df['typhoon_id']
        if 'typhoon_name' in df.columns:
            df['NAME'] = df['typhoon_name']
        if 'lat' in df.columns:
            df['LAT'] = df['lat']
        if 'lon' in df.columns:
            df['LON'] = df['lon']
        if 'wind' in df.columns:
            df['WMO_WIND'] = df['wind']
            df['USA_WIND'] = df['wind'] / 0.514444  # Convert m/s to knots for compatibility
        if 'pressure' in df.columns:
            df['WMO_PRES'] = df['pressure']
            df['USA_PRES'] = df['pressure']
        
        print(f"  Unique storms: {df['SID'].nunique()}")
    else:
        print(f"Interpolated file not found, using standard IBTrACS data")
        df = ibtracs_loader.load_ibtracs()
    
    # Filter typhoons
    storm_ids = ibtracs_loader.filter_typhoons(
        df,
        start_year=START_YEAR,
        end_year=END_YEAR,
        min_wind_speed=33.0,  # Strong tropical storm and above (~64 knots)
        min_duration_hours=48  # At least 48 hours
    )
    
    print(f"\n✓ Found {len(storm_ids)} strong typhoons ({START_YEAR}-{END_YEAR})")
    
    # Load ERA5 data for storms (if available)
    era5_loader, era5_datasets = load_era5_for_storms(storm_ids, df, ibtracs_loader)
    
    # Generate samples organized by year using systematic sliding windows
    samples_by_year, samples_by_storm = generate_samples_by_storm(
        ibtracs_loader=ibtracs_loader,
        df=df,
        storm_ids=storm_ids,
        past_timesteps=PAST_TIMESTEPS,
        future_timesteps=FUTURE_TIMESTEPS,
        stride=STRIDE,
        era5_loader=era5_loader,
        era5_datasets=era5_datasets
    )
    
    # Check if any samples were generated
    total_samples = sum(len(v) for v in samples_by_year.values())
    if total_samples == 0:
        print(f"\n⚠️  WARNING: No samples were generated (no ERA5 data available for any storms)")
        print(f"  Continuing anyway - training will proceed with available data only.")
        print(f"  Note: Training may not be possible without samples.")
    
    # Save samples by split
    save_samples_by_split(
        samples_by_year=samples_by_year,
        samples_by_storm=samples_by_storm,
        output_dir=OUTPUT_DIR,
        train_years=TRAIN_YEARS,
        val_years=VAL_YEARS,
        test_years=TEST_YEARS
    )
    
    print(f"\n{'='*80}")
    print("✓ DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\nYou can now train your model using:")
    print(f"  data_dir='{OUTPUT_DIR}'")
    print(f"\nThe dataset is split by year to avoid data leakage:")
    print(f"  - Training set uses {TRAIN_YEARS}")
    print(f"  - Validation set uses {VAL_YEARS}")
    print(f"  - Test set uses {TEST_YEARS}")
    print(f"\nThis ensures no typhoon appears in both training and test sets!")
    
    if era5_datasets:
        print(f"\n✓ Using ERA5 reanalysis data for {len(era5_datasets)} storms")
    else:
        print(f"\n⚠️  WARNING: No ERA5 datasets available")
        print(f"  Training will proceed with available data only.")
        print(f"  Generated {sum(len(v) for v in samples_by_year.values())} samples from IBTrACS data.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate typhoon dataset or check ERA5 data availability')
    parser.add_argument('--check-era5', action='store_true',
                        help='Check if all required ERA5 data files exist (does not generate dataset)')
    args = parser.parse_args()
    
    if args.check_era5:
        # Run ERA5 data check only
        print("="*80)
        print("ERA5 DATA AVAILABILITY CHECK")
        print("="*80)
        
        # Configuration
        START_YEAR = 2018
        END_YEAR = 2021
        
        # Initialize loader
        SCRIPT_DIR = Path(__file__).parent
        ibtracs_loader = IBTrACSLoader(
            data_dir=str(SCRIPT_DIR / "raw")
        )
        
        # Load IBTrACS data
        interpolated_file = SCRIPT_DIR / "raw" / "interpolated_typhoon_tracks_2018_2021.csv"
        if interpolated_file.exists():
            print(f"Loading interpolated tracks from {interpolated_file}")
            df = pd.read_csv(interpolated_file, low_memory=False)
            print(f"✓ Loaded {len(df)} interpolated records")
            
            # Map column names
            if 'typhoon_id' in df.columns:
                df['SID'] = df['typhoon_id']
            if 'typhoon_name' in df.columns:
                df['NAME'] = df['typhoon_name']
            if 'lat' in df.columns:
                df['LAT'] = df['lat']
            if 'lon' in df.columns:
                df['LON'] = df['lon']
            if 'wind' in df.columns:
                df['WMO_WIND'] = df['wind']
                df['USA_WIND'] = df['wind'] / 0.514444
            if 'pressure' in df.columns:
                df['WMO_PRES'] = df['pressure']
                df['USA_PRES'] = df['pressure']
        else:
            print(f"Interpolated file not found, using standard IBTrACS data")
            df = ibtracs_loader.load_ibtracs()
        
        # Filter typhoons
        storm_ids = ibtracs_loader.filter_typhoons(
            df,
            start_year=START_YEAR,
            end_year=END_YEAR,
            min_wind_speed=33.0,
            min_duration_hours=48
        )
        
        print(f"\nFound {len(storm_ids)} storms to check")
        
        # Run the check
        results = check_all_era5_data_exists(
            ibtracs_loader=ibtracs_loader,
            df=df,
            storm_ids=storm_ids,
            verbose=True
        )
        
        # Exit with appropriate code
        if results['all_exist']:
            print("\n✓ All ERA5 data is available. Ready to generate dataset.")
            exit(0)
        else:
            print("\n✗ Some ERA5 data is missing. Please download missing files before generating dataset.")
            exit(1)
    else:
        # Run normal dataset generation
        main()

