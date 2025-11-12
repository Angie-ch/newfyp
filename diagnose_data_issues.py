"""
Diagnostic Script for Data Quality Issues

This script helps identify and fix NaN problems in the preprocessed data.
"""

import numpy as np
import pickle
from pathlib import Path
import json
import sys

def check_nan_in_samples():
    """Check which samples and channels contain NaN"""
    
    print("="*80)
    print("CHECKING FOR NaN IN PREPROCESSED SAMPLES")
    print("="*80)
    
    data_dir = Path("data/processed/cases")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    sample_files = sorted(list(data_dir.glob("*.pkl")))
    
    if not sample_files:
        print(f"❌ No .pkl files found in {data_dir}")
        return
    
    print(f"Found {len(sample_files)} sample files\n")
    
    nan_samples = []
    nan_channels = {}
    
    for i, sample_file in enumerate(sample_files):
        try:
            with open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check past frames
            past_frames = data.get('past_frames')
            future_frames = data.get('future_frames')
            
            has_nan = False
            
            if past_frames is not None and np.isnan(past_frames).any():
                has_nan = True
                nan_samples.append(sample_file.name)
                
                # Find which channels have NaN
                for ch in range(past_frames.shape[1]):
                    if np.isnan(past_frames[:, ch]).any():
                        if ch not in nan_channels:
                            nan_channels[ch] = []
                        nan_channels[ch].append(sample_file.name)
            
            if future_frames is not None and np.isnan(future_frames).any():
                has_nan = True
                if sample_file.name not in nan_samples:
                    nan_samples.append(sample_file.name)
            
            if has_nan:
                print(f"❌ {sample_file.name}: Contains NaN")
            elif i % 20 == 0:
                print(f"✓ Checked {i+1}/{len(sample_files)} samples...")
                
        except Exception as e:
            print(f"❌ Error loading {sample_file.name}: {e}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Total samples: {len(sample_files)}")
    print(f"Samples with NaN: {len(nan_samples)} ({100*len(nan_samples)/len(sample_files):.1f}%)")
    
    if nan_channels:
        print(f"\nChannels containing NaN:")
        for ch, files in sorted(nan_channels.items()):
            print(f"  Channel {ch}: {len(files)} samples")
    
    return nan_samples, nan_channels


def check_statistics():
    """Check normalization statistics file"""
    
    print("\n" + "="*80)
    print("CHECKING NORMALIZATION STATISTICS")
    print("="*80)
    
    stats_file = Path("data/processed/normalization_stats.pkl")
    
    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        return None
    
    try:
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        
        print(f"✓ Loaded statistics from {stats_file}")
        print(f"\nStatistics keys: {list(stats.keys())}")
        
        if 'mean' in stats:
            mean = stats['mean']
            print(f"\nMean shape: {mean.shape}")
            print(f"Mean contains NaN: {np.isnan(mean).any()}")
            if np.isnan(mean).any():
                nan_indices = np.where(np.isnan(mean))[0]
                print(f"  NaN at indices: {nan_indices}")
        
        if 'std' in stats:
            std = stats['std']
            print(f"\nStd shape: {std.shape}")
            print(f"Std contains NaN: {np.isnan(std).any()}")
            if np.isnan(std).any():
                nan_indices = np.where(np.isnan(std))[0]
                print(f"  NaN at indices: {nan_indices}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error loading statistics: {e}")
        return None


def check_json_statistics():
    """Check JSON statistics file"""
    
    print("\n" + "="*80)
    print("CHECKING JSON STATISTICS")
    print("="*80)
    
    stats_file = Path("data/processed/statistics.json")
    
    if not stats_file.exists():
        print(f"❌ JSON statistics file not found: {stats_file}")
        return None
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"✓ Loaded JSON statistics from {stats_file}")
        print(json.dumps(stats, indent=2))
        
        # Check for NaN (which appears as NaN in JSON, not null)
        has_nan = False
        for key in ['mean', 'std', 'min', 'max']:
            if key in stats:
                try:
                    if stats[key] is None or (isinstance(stats[key], float) and np.isnan(stats[key])):
                        print(f"❌ {key} is NaN")
                        has_nan = True
                except:
                    # In JSON, NaN might appear as string "NaN"
                    if str(stats[key]) == "NaN":
                        print(f"❌ {key} is NaN (string)")
                        has_nan = True
        
        if not has_nan:
            print("✓ No NaN values detected in JSON statistics")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error loading JSON statistics: {e}")
        return None


def recompute_statistics():
    """Recompute statistics with proper NaN handling"""
    
    print("\n" + "="*80)
    print("RECOMPUTING STATISTICS WITH NaN HANDLING")
    print("="*80)
    
    data_dir = Path("data/processed/cases")
    sample_files = sorted(list(data_dir.glob("*.pkl")))
    
    if not sample_files:
        print("❌ No sample files found")
        return
    
    print(f"Processing {len(sample_files)} samples...")
    
    all_frames = []
    valid_samples = 0
    
    for sample_file in sample_files:
        try:
            with open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            past_frames = data.get('past_frames')
            future_frames = data.get('future_frames')
            
            # Skip if contains NaN
            if past_frames is not None and not np.isnan(past_frames).any():
                all_frames.append(past_frames)
                valid_samples += 1
            
            if future_frames is not None and not np.isnan(future_frames).any():
                all_frames.append(future_frames)
                
        except Exception as e:
            print(f"❌ Error processing {sample_file.name}: {e}")
    
    if not all_frames:
        print("❌ No valid frames found")
        return
    
    print(f"\n✓ Found {valid_samples} valid samples")
    
    # Concatenate and compute statistics
    all_frames = np.concatenate(all_frames, axis=0)  # (N, C, H, W)
    
    print(f"Total frames shape: {all_frames.shape}")
    
    # Compute channel-wise statistics
    mean = np.mean(all_frames, axis=(0, 2, 3))  # (C,)
    std = np.std(all_frames, axis=(0, 2, 3))    # (C,)
    
    print(f"\nComputed statistics:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std shape: {std.shape}")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    # Check for NaN
    if np.isnan(mean).any() or np.isnan(std).any():
        print("\n❌ WARNING: Computed statistics still contain NaN!")
        print("   This means there's an issue with the data itself.")
        return
    
    # Save new statistics
    new_stats = {
        'mean': mean,
        'std': std,
        'n_samples': valid_samples
    }
    
    output_file = Path("data/processed/normalization_stats_fixed.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(new_stats, f)
    
    print(f"\n✓ Saved fixed statistics to {output_file}")
    print("\nTo use these statistics, rename:")
    print(f"  mv {output_file} data/processed/normalization_stats.pkl")
    
    return new_stats


def analyze_channel_ranges():
    """Analyze value ranges for each channel"""
    
    print("\n" + "="*80)
    print("ANALYZING CHANNEL VALUE RANGES")
    print("="*80)
    
    data_dir = Path("data/processed/cases")
    sample_files = sorted(list(data_dir.glob("*.pkl")))[:10]  # Sample first 10
    
    channel_mins = {}
    channel_maxs = {}
    
    for sample_file in sample_files:
        try:
            with open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            past_frames = data.get('past_frames')
            
            if past_frames is not None:
                for ch in range(past_frames.shape[1]):
                    channel_data = past_frames[:, ch, :, :]
                    
                    # Skip NaN
                    if not np.isnan(channel_data).all():
                        ch_min = np.nanmin(channel_data)
                        ch_max = np.nanmax(channel_data)
                        
                        if ch not in channel_mins:
                            channel_mins[ch] = ch_min
                            channel_maxs[ch] = ch_max
                        else:
                            channel_mins[ch] = min(channel_mins[ch], ch_min)
                            channel_maxs[ch] = max(channel_maxs[ch], ch_max)
                            
        except Exception as e:
            continue
    
    print(f"\nChannel value ranges (from first 10 samples):")
    print(f"{'Channel':<10} {'Min':<15} {'Max':<15} {'Range':<15}")
    print("-" * 60)
    
    for ch in sorted(channel_mins.keys()):
        ch_min = channel_mins[ch]
        ch_max = channel_maxs[ch]
        ch_range = ch_max - ch_min
        print(f"{ch:<10} {ch_min:<15.4f} {ch_max:<15.4f} {ch_range:<15.4f}")


def main():
    """Run all diagnostics"""
    
    print("\n" + "="*80)
    print("TYPHOON PREDICTION DATA DIAGNOSTICS")
    print("="*80 + "\n")
    
    # Check samples
    nan_samples, nan_channels = check_nan_in_samples()
    
    # Check statistics files
    pkl_stats = check_statistics()
    json_stats = check_json_statistics()
    
    # Analyze channel ranges
    analyze_channel_ranges()
    
    # Offer to recompute
    if nan_samples:
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        print(f"\n⚠️  Found {len(nan_samples)} samples with NaN values")
        print("\nOptions:")
        print("  1. Remove NaN samples and recompute statistics")
        print("  2. Fix the preprocessing pipeline to avoid NaN")
        print("  3. Investigate root cause of NaN values")
        
        response = input("\nRecompute statistics excluding NaN samples? (y/n): ")
        if response.lower() == 'y':
            recompute_statistics()
    else:
        print("\n" + "="*80)
        print("✅ NO DATA ISSUES FOUND")
        print("="*80)


if __name__ == "__main__":
    main()

