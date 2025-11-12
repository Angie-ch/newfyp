"""
Verify that the temporal split is correct and no data leakage exists

This script checks:
1. No typhoon appears in multiple splits (train/val/test)
2. Years are correctly separated
3. Sample statistics are reasonable
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle


def extract_info_from_filename(filename: str):
    """
    Extract year and storm_id from filename
    Format: YEAR_STORMID_sINDEX.npz
    Example: 2021_2021WP01_s0.npz
    """
    parts = filename.replace('.npz', '').replace('.pkl', '').split('_')
    year = int(parts[0])
    
    # Storm ID might contain underscores
    storm_id_parts = parts[1:-1]  # Everything except year and sample index
    storm_id = '_'.join(storm_id_parts)
    
    return year, storm_id


def analyze_split_directory(split_dir: Path, split_name: str):
    """Analyze a single split directory"""
    cases_dir = split_dir / 'cases'
    
    if not cases_dir.exists():
        print(f"  ⚠️  Cases directory not found: {cases_dir}")
        return None, None, None
    
    files = list(cases_dir.glob("*.npz")) + list(cases_dir.glob("*.pkl"))
    files = [f for f in files if not f.name.startswith('.') and not f.name.startswith('._')]
    
    if not files:
        print(f"  ⚠️  No data files found in {cases_dir}")
        return None, None, None
    
    years = set()
    storm_ids = set()
    
    for file in files:
        try:
            year, storm_id = extract_info_from_filename(file.name)
            years.add(year)
            storm_ids.add(storm_id)
        except Exception as e:
            print(f"  ⚠️  Could not parse filename: {file.name} - {e}")
    
    print(f"\n{split_name.upper()} Split:")
    print(f"  📁 Files: {len(files)}")
    print(f"  📅 Years: {sorted(years)}")
    print(f"  🌀 Unique typhoons: {len(storm_ids)}")
    print(f"  📊 Samples per typhoon: {len(files) / len(storm_ids):.1f} (avg)")
    
    return files, years, storm_ids


def verify_no_leakage(train_storms, val_storms, test_storms):
    """Verify that no typhoon appears in multiple splits"""
    print("\n" + "="*80)
    print("DATA LEAKAGE CHECK")
    print("="*80)
    
    # Check for overlaps
    train_val_overlap = train_storms & val_storms
    train_test_overlap = train_storms & test_storms
    val_test_overlap = val_storms & test_storms
    
    has_leakage = False
    
    if train_val_overlap:
        print(f"\n❌ LEAKAGE DETECTED: {len(train_val_overlap)} typhoons appear in both TRAIN and VAL!")
        print(f"   Overlapping storms: {list(train_val_overlap)[:5]}...")
        has_leakage = True
    else:
        print("\n✅ No overlap between TRAIN and VAL")
    
    if train_test_overlap:
        print(f"\n❌ LEAKAGE DETECTED: {len(train_test_overlap)} typhoons appear in both TRAIN and TEST!")
        print(f"   Overlapping storms: {list(train_test_overlap)[:5]}...")
        has_leakage = True
    else:
        print("✅ No overlap between TRAIN and TEST")
    
    if val_test_overlap:
        print(f"\n❌ LEAKAGE DETECTED: {len(val_test_overlap)} typhoons appear in both VAL and TEST!")
        print(f"   Overlapping storms: {list(val_test_overlap)[:5]}...")
        has_leakage = True
    else:
        print("✅ No overlap between VAL and TEST")
    
    if not has_leakage:
        print("\n" + "🎉 " * 20)
        print("✅ NO DATA LEAKAGE DETECTED - SPLIT IS CORRECT!")
        print("🎉 " * 20)
    
    return not has_leakage


def verify_year_separation(train_years, val_years, test_years):
    """Verify that years are correctly separated"""
    print("\n" + "="*80)
    print("YEAR SEPARATION CHECK")
    print("="*80)
    
    # Check for overlaps
    train_val_year_overlap = train_years & val_years
    train_test_year_overlap = train_years & test_years
    val_test_year_overlap = val_years & test_years
    
    has_year_leakage = False
    
    if train_val_year_overlap:
        print(f"\n❌ YEAR LEAKAGE: Years {train_val_year_overlap} appear in both TRAIN and VAL!")
        has_year_leakage = True
    else:
        print("\n✅ No year overlap between TRAIN and VAL")
    
    if train_test_year_overlap:
        print(f"\n❌ YEAR LEAKAGE: Years {train_test_year_overlap} appear in both TRAIN and TEST!")
        has_year_leakage = True
    else:
        print("✅ No year overlap between TRAIN and TEST")
    
    if val_test_year_overlap:
        print(f"\n❌ YEAR LEAKAGE: Years {val_test_year_overlap} appear in both VAL and TEST!")
        has_year_leakage = True
    else:
        print("✅ No year overlap between VAL and TEST")
    
    return not has_year_leakage


def load_and_verify_sample(file_path: Path):
    """Load a sample and verify its contents"""
    try:
        data = np.load(file_path, allow_pickle=True)
        
        info = {
            'has_past_frames': 'past_frames' in data,
            'has_future_frames': 'future_frames' in data,
            'has_track': 'track_past' in data and 'track_future' in data,
            'has_intensity': 'intensity_past' in data and 'intensity_future' in data,
            'has_year': 'year' in data,
            'has_storm_id': 'storm_id' in data
        }
        
        if info['has_past_frames']:
            past_shape = data['past_frames'].shape
            info['past_shape'] = past_shape
            info['has_nan'] = np.isnan(data['past_frames']).any()
        
        if info['has_future_frames']:
            future_shape = data['future_frames'].shape
            info['future_shape'] = future_shape
            info['has_nan'] = info.get('has_nan', False) or np.isnan(data['future_frames']).any()
        
        return info
    except Exception as e:
        return {'error': str(e)}


def main():
    """Main verification function"""
    print("="*80)
    print("TEMPORAL SPLIT VERIFICATION")
    print("="*80)
    
    # Default data directory
    data_dir = Path("data/processed_temporal_split")
    
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("\nPlease run 'python generate_data_by_year.py' first to generate the dataset.")
        return
    
    print(f"\nData directory: {data_dir}")
    
    # Analyze each split
    train_files, train_years, train_storms = analyze_split_directory(data_dir / 'train', 'train')
    val_files, val_years, val_storms = analyze_split_directory(data_dir / 'val', 'val')
    test_files, test_years, test_storms = analyze_split_directory(data_dir / 'test', 'test')
    
    if train_storms is None or val_storms is None or test_storms is None:
        print("\n❌ Could not analyze all splits. Please check the data directory structure.")
        return
    
    # Verify no data leakage
    no_leakage = verify_no_leakage(train_storms, val_storms, test_storms)
    
    # Verify year separation
    year_separation_ok = verify_year_separation(train_years, val_years, test_years)
    
    # Check metadata
    metadata_file = data_dir / 'dataset_metadata.pkl'
    if metadata_file.exists():
        print("\n" + "="*80)
        print("METADATA CHECK")
        print("="*80)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"\n✅ Metadata file found")
        print(f"  Train years: {metadata.get('train_years')}")
        print(f"  Val years: {metadata.get('val_years')}")
        print(f"  Test years: {metadata.get('test_years')}")
        print(f"  Train samples: {metadata.get('n_train')}")
        print(f"  Val samples: {metadata.get('n_val')}")
        print(f"  Test samples: {metadata.get('n_test')}")
    else:
        print(f"\n⚠️  Metadata file not found: {metadata_file}")
    
    # Sample data quality check
    print("\n" + "="*80)
    print("SAMPLE DATA QUALITY CHECK")
    print("="*80)
    
    if train_files:
        print(f"\nChecking random sample from train split...")
        sample_file = train_files[0]
        sample_info = load_and_verify_sample(sample_file)
        
        if 'error' in sample_info:
            print(f"❌ Error loading sample: {sample_info['error']}")
        else:
            print(f"✅ Sample loaded successfully")
            print(f"  Has past frames: {sample_info.get('has_past_frames')}")
            print(f"  Has future frames: {sample_info.get('has_future_frames')}")
            print(f"  Has track data: {sample_info.get('has_track')}")
            print(f"  Has intensity data: {sample_info.get('has_intensity')}")
            print(f"  Has year metadata: {sample_info.get('has_year')}")
            print(f"  Has storm ID: {sample_info.get('has_storm_id')}")
            if 'past_shape' in sample_info:
                print(f"  Past frames shape: {sample_info['past_shape']}")
            if 'future_shape' in sample_info:
                print(f"  Future frames shape: {sample_info['future_shape']}")
            if sample_info.get('has_nan'):
                print(f"  ⚠️  Contains NaN values")
            else:
                print(f"  ✅ No NaN values")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if no_leakage and year_separation_ok:
        print("\n✅✅✅ DATASET IS CORRECTLY SPLIT - SAFE TO USE! ✅✅✅")
        print("\nYou can now train your model with confidence that:")
        print("  • No typhoon appears in multiple splits")
        print("  • Years are properly separated")
        print("  • No data leakage exists")
    else:
        print("\n❌❌❌ ISSUES DETECTED - DO NOT USE THIS DATASET! ❌❌❌")
        print("\nPlease regenerate the dataset using:")
        print("  python generate_data_by_year.py")


if __name__ == "__main__":
    main()

