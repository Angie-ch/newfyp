#!/usr/bin/env python3
"""
Preprocess ERA5 and IBTrACS data with 1-hour interpolation
Creates training-ready samples from 2018-2021 ERA5 data
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.typhoon_preprocessor import TyphoonPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def main():
    """Main preprocessing pipeline"""
    print_header("ERA5 + IBTrACS PREPROCESSING PIPELINE")
    print(f"Start Time: {datetime.now()}")
    
    # Configuration
    config = {
        'era5_dir': 'data/era5',
        'ibtracs_file': 'data/raw/ibtracs_wp.csv',
        'output_dir': 'data/processed',
        'start_date': '2018-01-01',
        'end_date': '2021-12-31',
        'input_frames': 8,       # 8 hours of past data (1-hour intervals)
        'output_frames': 12,     # 12 hours of future prediction
        'time_interval_hours': 1,  # 1-hour interpolation
        'min_intensity': 17.0,   # Minimum wind speed (m/s) - Tropical Storm threshold
        'max_samples': None,     # Process all available storms
    }
    
    print("\nConfiguration:")
    print(f"  ERA5 Directory: {config['era5_dir']}")
    print(f"  IBTrACS File: {config['ibtracs_file']}")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Time Range: {config['start_date']} to {config['end_date']}")
    print(f"  Time Interval: {config['time_interval_hours']} hour(s)")
    print(f"  Input Frames: {config['input_frames']} (past)")
    print(f"  Output Frames: {config['output_frames']} (future)")
    print(f"  Min Intensity: {config['min_intensity']} m/s")
    
    # Check if files exist
    era5_dir = Path(config['era5_dir'])
    ibtracs_file = Path(config['ibtracs_file'])
    
    if not era5_dir.exists():
        print(f"\n✗ ERROR: ERA5 directory not found: {era5_dir}")
        print("  Please ensure ERA5 data is in the correct location")
        sys.exit(1)
    
    if not ibtracs_file.exists():
        print(f"\n✗ ERROR: IBTrACS file not found: {ibtracs_file}")
        print("  Please download IBTrACS Western Pacific data")
        sys.exit(1)
    
    # Count ERA5 files
    era5_files = list(era5_dir.rglob("*.nc"))
    print(f"\n✓ Found {len(era5_files)} ERA5 files")
    
    # Initialize preprocessor
    print_header("INITIALIZING PREPROCESSOR")
    
    preprocessor = TyphoonPreprocessor(
        era5_dir=str(era5_dir),
        ibtracs_file=str(ibtracs_file),
        output_dir=config['output_dir'],
        input_frames=config['input_frames'],
        output_frames=config['output_frames'],
        time_interval_hours=config['time_interval_hours']
    )
    
    print("✓ Preprocessor initialized")
    
    # Get list of storms
    print_header("FINDING TYPHOON CASES")
    
    storm_ids = preprocessor.ibtracs_processor.get_storms_in_timerange(
        config['start_date'],
        config['end_date'],
        config['min_intensity']
    )
    
    print(f"✓ Found {len(storm_ids)} typhoons matching criteria")
    
    if len(storm_ids) == 0:
        print("\n✗ No storms found in the specified time range!")
        print("  Check your IBTrACS data and date range")
        sys.exit(1)
    
    # Show sample storms
    print("\nSample storms:")
    df = preprocessor.ibtracs_processor.data
    for i, sid in enumerate(storm_ids[:10]):
        storm_data = df[df['SID'] == sid].iloc[0]
        name = storm_data.get('NAME', 'UNNAMED')
        year = pd.to_datetime(storm_data['ISO_TIME']).year
        print(f"  {i+1}. {name} ({sid}) - {year}")
    if len(storm_ids) > 10:
        print(f"  ... and {len(storm_ids) - 10} more")
    
    # Compute global statistics
    print_header("COMPUTING GLOBAL STATISTICS")
    
    print("Computing normalization statistics from sample of storms...")
    print("(This may take a while as ERA5 files are being loaded)")
    
    stats = preprocessor.compute_global_statistics(
        storm_ids=storm_ids,
        num_samples=min(50, len(storm_ids))
    )
    
    if stats is None:
        print("✗ Failed to compute statistics")
        sys.exit(1)
    
    print("✓ Statistics computed successfully")
    print(f"  Mean range: [{stats['mean'].min():.3f}, {stats['mean'].max():.3f}]")
    print(f"  Std range: [{stats['std'].min():.3f}, {stats['std'].max():.3f}]")
    
    # Process all storms
    print_header("PROCESSING TYPHOON CASES")
    
    if config['max_samples']:
        storm_ids = storm_ids[:config['max_samples']]
        print(f"Processing first {config['max_samples']} storms")
    else:
        print(f"Processing all {len(storm_ids)} storms")
    
    print("\nThis will take a while - processing ERA5 data for each timestep...")
    print("Progress:")
    
    processed_files = preprocessor.process_all_storms(
        start_date=config['start_date'],
        end_date=config['end_date'],
        min_intensity=config['min_intensity'],
        max_samples=config['max_samples']
    )
    
    # Create metadata
    print_header("CREATING METADATA")
    
    metadata = []
    for i, pkl_file in enumerate(processed_files):
        import pickle
        with open(pkl_file, 'rb') as f:
            case_data = pickle.load(f)
        
        metadata.append({
            'case_id': i,
            'storm_id': case_data['storm_id'],
            'max_past_intensity': float(case_data['input_intensity'].max()),
            'max_future_intensity': float(case_data['target_intensity'].max()),
            'n_frames': config['input_frames'] + config['output_frames'],
            'time_interval_hours': config['time_interval_hours']
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_file = Path(config['output_dir']) / 'metadata.csv'
    metadata_df.to_csv(metadata_file, index=False)
    print(f"✓ Saved metadata: {metadata_file}")
    
    # Save configuration
    config_output = {
        'creation_date': datetime.now().isoformat(),
        'data_source': 'ERA5 + IBTrACS Western Pacific',
        'time_range': f"{config['start_date']} to {config['end_date']}",
        'time_interval_hours': config['time_interval_hours'],
        'input_frames': config['input_frames'],
        'output_frames': config['output_frames'],
        'total_frames': config['input_frames'] + config['output_frames'],
        'n_storms': len(storm_ids),
        'n_processed': len(processed_files),
        'min_intensity_threshold': config['min_intensity'],
        'normalization_stats': {
            'mean': stats['mean'].tolist(),
            'std': stats['std'].tolist()
        }
    }
    
    config_file = Path(config['output_dir']) / 'preprocessing_config.json'
    with open(config_file, 'w') as f:
        json.dump(config_output, f, indent=2)
    print(f"✓ Saved configuration: {config_file}")
    
    # Summary
    print_header("PREPROCESSING COMPLETE")
    print(f"End Time: {datetime.now()}")
    print(f"\n✓ Successfully processed {len(processed_files)} typhoon cases")
    print(f"\nOutput:")
    print(f"  Processed cases: {config['output_dir']}/")
    print(f"  Metadata: {metadata_file}")
    print(f"  Configuration: {config_file}")
    print(f"\nDataset Info:")
    print(f"  Time Resolution: {config['time_interval_hours']} hour")
    print(f"  Input Sequence: {config['input_frames']} frames (past)")
    print(f"  Output Sequence: {config['output_frames']} frames (future)")
    print(f"  Total Duration: {config['input_frames'] + config['output_frames']} hours per case")
    print("\nNext steps:")
    print("  1. Train autoencoder: python train_autoencoder.py")
    print("  2. Train diffusion: python train_diffusion.py")
    print("  3. Run inference: python inference.py")
    print("=" * 80)


if __name__ == '__main__':
    main()


