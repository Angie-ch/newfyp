"""
Data Preprocessing Script

Usage:
    python preprocess_data.py --era5_dir data/raw/era5 \
                              --ibtracs data/raw/ibtracs.csv \
                              --output data/processed \
                              --start_date 2015-01-01 \
                              --end_date 2020-12-31
"""

import argparse
from pathlib import Path

from data.preprocessing import TyphoonPreprocessor


def main(args):
    """Main preprocessing function"""
    
    print("="*80)
    print("Typhoon Data Preprocessing")
    print("="*80)
    
    # Create preprocessor
    preprocessor = TyphoonPreprocessor(
        era5_dir=args.era5_dir,
        ibtracs_file=args.ibtracs,
        output_dir=args.output,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        time_interval_hours=args.time_interval
    )
    
    # Compute global statistics if requested
    if args.compute_stats:
        print("\nComputing global statistics...")
        
        # Get all storms in time range
        storm_ids = preprocessor.ibtracs_processor.get_storms_in_timerange(
            args.start_date,
            args.end_date,
            min_intensity=args.min_intensity
        )
        
        stats = preprocessor.compute_global_statistics(
            storm_ids,
            num_samples=args.stats_samples
        )
        
        print(f"  Statistics computed from {args.stats_samples} samples")
    
    # Process all storms
    print("\nProcessing storms...")
    processed_files = preprocessor.process_all_storms(
        start_date=args.start_date,
        end_date=args.end_date,
        min_intensity=args.min_intensity,
        max_samples=args.max_samples
    )
    
    print(f"\nProcessed {len(processed_files)} typhoon cases")
    print(f"Output saved to: {args.output}")
    
    print("\n" + "="*80)
    print("Preprocessing complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Typhoon Data')
    
    # Data paths
    parser.add_argument('--era5_dir', type=str, required=True,
                        help='Directory containing ERA5 netCDF files')
    parser.add_argument('--ibtracs', type=str, required=True,
                        help='Path to IBTrACS CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed data')
    
    # Time range
    parser.add_argument('--start_date', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    
    # Processing parameters
    parser.add_argument('--input_frames', type=int, default=12,
                        help='Number of input frames')
    parser.add_argument('--output_frames', type=int, default=8,
                        help='Number of output frames')
    parser.add_argument('--time_interval', type=int, default=6,
                        help='Time interval in hours')
    parser.add_argument('--min_intensity', type=float, default=17.0,
                        help='Minimum wind speed threshold (m/s)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    # Statistics
    parser.add_argument('--compute_stats', action='store_true',
                        help='Compute global normalization statistics')
    parser.add_argument('--stats_samples', type=int, default=100,
                        help='Number of samples for statistics computation')
    
    args = parser.parse_args()
    
    main(args)

