"""
Compute normalization statistics for Track and Intensity data
"""

import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def compute_statistics(data_dir: str, split: str = 'train'):
    """
    Compute statistics for Track and Intensity normalization
    
    Args:
        data_dir: Root directory containing processed data
        split: 'train', 'val', or 'test' to compute stats from
    """
    data_dir = Path(data_dir)
    cases_dir = data_dir / split / 'cases'
    
    if not cases_dir.exists():
        cases_dir = data_dir / split
    
    if not cases_dir.exists():
        raise ValueError(f"Data directory not found: {cases_dir}")
    
    # Find all .npz files
    all_files = list(cases_dir.glob('*.npz'))
    sample_files = sorted([
        f for f in all_files 
        if not f.name.startswith('._')
    ])
    
    if len(sample_files) == 0:
        raise ValueError(f"No .npz files found in {cases_dir}")
    
    print(f"Computing statistics from {len(sample_files)} files in {cases_dir}")
    
    # Accumulate statistics
    track_values = []
    intensity_values = []
    
    for sample_file in tqdm(sample_files, desc="Loading samples"):
        try:
            data = np.load(sample_file, allow_pickle=True)
            
            # Collect track values (lat, lon)
            track_past = data['track_past'].astype(np.float32)
            track_future = data['track_future'].astype(np.float32)
            track_all = np.concatenate([track_past, track_future], axis=0)
            track_values.append(track_all)
            
            # Collect intensity values (filter out NaN and invalid values)
            intensity_past = data['intensity_past'].astype(np.float32)
            intensity_future = data['intensity_future'].astype(np.float32)
            intensity_all = np.concatenate([intensity_past, intensity_future])
            # Filter out NaN, Inf, and negative values
            intensity_all = intensity_all[np.isfinite(intensity_all) & (intensity_all >= 0)]
            if len(intensity_all) > 0:
                intensity_values.append(intensity_all)
        except Exception as e:
            print(f"Warning: Error loading {sample_file.name}: {e}")
            continue
    
    # Compute statistics
    if len(track_values) == 0:
        raise ValueError("No valid track data found")
    
    track_all = np.concatenate(track_values, axis=0)  # (N, 2)
    track_mean = np.mean(track_all, axis=0).tolist()  # [lat_mean, lon_mean]
    track_std = np.std(track_all, axis=0).tolist()    # [lat_std, lon_std]
    
    if len(intensity_values) == 0:
        print("Warning: No valid intensity data found, using default values")
        intensity_mean = 10.0
        intensity_std = 10.0
    else:
        intensity_all = np.concatenate(intensity_values)  # (N,)
        # Filter out any remaining invalid values
        intensity_all = intensity_all[np.isfinite(intensity_all) & (intensity_all >= 0)]
        if len(intensity_all) == 0:
            print("Warning: All intensity values are invalid, using default values")
            intensity_mean = 10.0
            intensity_std = 10.0
        else:
            intensity_mean = float(np.mean(intensity_all))
            intensity_std = float(np.std(intensity_all))
            # Ensure std is not too small
            if intensity_std < 1e-6:
                intensity_std = 10.0
    
    # Print results
    print("\n" + "="*60)
    print("COMPUTED STATISTICS")
    print("="*60)
    print(f"Track (lat, lon):")
    print(f"  Mean: {track_mean}")
    print(f"  Std:  {track_std}")
    print(f"\nIntensity (wind speed):")
    print(f"  Mean: {intensity_mean:.4f}")
    print(f"  Std:  {intensity_std:.4f}")
    print("="*60)
    
    # Load existing statistics.json if it exists
    stats_file = data_dir / 'statistics.json'
    if not stats_file.exists() and split == 'train':
        stats_file = data_dir.parent / 'statistics.json'
    
    stats = {}
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\nLoaded existing statistics from {stats_file}")
    else:
        print(f"\nNo existing statistics file found, creating new one")
    
    # Update with new statistics
    stats['track_mean'] = track_mean
    stats['track_std'] = track_std
    stats['intensity_mean'] = intensity_mean
    stats['intensity_std'] = intensity_std
    
    # Save updated statistics
    output_file = data_dir / 'statistics.json'
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved statistics to {output_file}")
    
    return stats


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compute_statistics.py <data_dir> [split]")
        print("Example: python compute_statistics.py data/processed_temporal_split train")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else 'train'
    
    compute_statistics(data_dir, split)

