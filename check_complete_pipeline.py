"""
Complete Pipeline Check Script

This script checks all components of the typhoon prediction pipeline:
1. Data format and integrity
2. IBTrACS encoding/decoding
3. Autoencoder (if checkpoint exists)
4. Diffusion model (if checkpoint exists)
5. Visualization capabilities
"""

import torch
import numpy as np
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_data_format():
    """Check that processed data has correct format"""
    logger.info("\n" + "="*60)
    logger.info("1. CHECKING DATA FORMAT")
    logger.info("="*60)
    
    cases_dir = Path('data/processed/cases')
    if not cases_dir.exists():
        logger.error(f"✗ Data directory not found: {cases_dir}")
        return False
    
    # Get valid samples (exclude macOS temp files)
    samples = [f for f in cases_dir.glob('*.npz') if not f.name.startswith('._')]
    
    if len(samples) == 0:
        logger.error("✗ No data files found")
        return False
    
    logger.info(f"✓ Found {len(samples)} data files")
    
    # Check first few samples
    issues = []
    for i, sample_file in enumerate(samples[:3]):
        logger.info(f"\n  Checking: {sample_file.name}")
        try:
            data = np.load(sample_file, allow_pickle=True)
            
            required_keys = ['past_frames', 'future_frames', 'track_past', 'track_future']
            missing = [k for k in required_keys if k not in data]
            if missing:
                issues.append(f"{sample_file.name}: Missing keys {missing}")
                continue
            
            # Check shapes
            past_frames = data['past_frames']
            future_frames = data['future_frames']
            track_past = data['track_past']
            track_future = data['track_future']
            
            logger.info(f"    past_frames: {past_frames.shape}")
            logger.info(f"    future_frames: {future_frames.shape}")
            logger.info(f"    track_past: {track_past.shape}")
            logger.info(f"    track_future: {track_future.shape}")
            
            # Check for NaN in frames
            if np.isnan(past_frames).any():
                issues.append(f"{sample_file.name}: past_frames contains NaN")
            if np.isnan(future_frames).any():
                issues.append(f"{sample_file.name}: future_frames contains NaN")
            
            # Check track coordinates
            if np.isnan(track_past).any() or np.isnan(track_future).any():
                issues.append(f"{sample_file.name}: tracks contain NaN")
            else:
                # Check coordinate ranges - CRITICAL CHECK
                lat_past = track_past[:, 0]
                lon_past = track_past[:, 1]
                lat_future = track_future[:, 0]
                lon_future = track_future[:, 1]
                
                logger.info(f"    Latitude range (past): [{lat_past.min():.2f}, {lat_past.max():.2f}]")
                logger.info(f"    Longitude range (past): [{lon_past.min():.2f}, {lon_past.max():.2f}]")
                
                # Valid ranges for Western Pacific typhoons:
                # Latitude: ~5°N to ~45°N
                # Longitude: ~100°E to ~180°E
                if lat_past.min() < 0 or lat_past.max() > 50:
                    if lat_past.min() > 100:  # Likely swapped
                        issues.append(f"{sample_file.name}: ⚠️  COORDINATES SWAPPED! Lat values look like lon ({lat_past.min():.1f}-{lat_past.max():.1f})")
                    else:
                        issues.append(f"{sample_file.name}: Latitude out of range [{lat_past.min():.2f}, {lat_past.max():.2f}]")
                
                if lon_past.min() < 100 or lon_past.max() > 200:
                    if lon_past.max() < 50:  # Likely swapped
                        issues.append(f"{sample_file.name}: ⚠️  COORDINATES SWAPPED! Lon values look like lat ({lon_past.min():.1f}-{lon_past.max():.1f})")
                    else:
                        issues.append(f"{sample_file.name}: Longitude out of range [{lon_past.min():.2f}, {lon_past.max():.2f}]")
            
        except Exception as e:
            issues.append(f"{sample_file.name}: Error loading - {e}")
    
    if issues:
        logger.error("\n  ✗ Issues found:")
        for issue in issues:
            logger.error(f"    - {issue}")
        return False
    else:
        logger.info("\n  ✓ All data checks passed")
        return True


def check_ibtracs_encoding():
    """Check IBTrACS encoding/decoding"""
    logger.info("\n" + "="*60)
    logger.info("2. CHECKING IBTRACS ENCODING/DECODING")
    logger.info("="*60)
    
    try:
        from data.utils.ibtracs_encoding import encode_track_as_channels, decode_track_from_channels
        
        # Test with sample coordinates (Western Pacific)
        test_lat = torch.tensor([[20.0, 22.5, 25.0]])  # (1, 3) - 3 timesteps
        test_lon = torch.tensor([[130.0, 132.0, 134.0]])
        test_track = torch.stack([test_lat, test_lon], dim=-1)  # (1, 3, 2) [lat, lon]
        test_intensity = torch.tensor([[30.0, 35.0, 40.0]])
        
        logger.info(f"  Input track shape: {test_track.shape}")
        logger.info(f"  Input coordinates: lat={test_lat[0].tolist()}, lon={test_lon[0].tolist()}")
        
        # Encode
        encoded = encode_track_as_channels(
            test_track, test_intensity,
            image_size=(64, 64),
            grid_range=(20.0, 20.0),
            normalize=True
        )
        logger.info(f"  Encoded shape: {encoded.shape}")
        
        # Decode (returns tuple: track, intensity, pressure)
        decoded_track, decoded_intensity, decoded_pressure = decode_track_from_channels(encoded, grid_range=(20.0, 20.0))
        logger.info(f"  Decoded track shape: {decoded_track.shape}")
        logger.info(f"  Decoded coordinates: lat={decoded_track[0, :, 0].tolist()}, lon={decoded_track[0, :, 1].tolist()}")
        
        # Check accuracy
        error_lat = torch.abs(test_track[..., 0] - decoded_track[..., 0]).max().item()
        error_lon = torch.abs(test_track[..., 1] - decoded_track[..., 1]).max().item()
        
        logger.info(f"  Max latitude error: {error_lat:.6f}°")
        logger.info(f"  Max longitude error: {error_lon:.6f}°")
        
        if error_lat < 0.01 and error_lon < 0.01:
            logger.info("  ✓ Encoding/decoding works correctly")
            return True
        else:
            logger.error(f"  ✗ Large encoding error (lat: {error_lat:.4f}, lon: {error_lon:.4f})")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset():
    """Check TyphoonDataset class"""
    logger.info("\n" + "="*60)
    logger.info("3. CHECKING DATASET LOADER")
    logger.info("="*60)
    
    try:
        from data.datasets.typhoon_dataset import TyphoonDataset
        
        dataset = TyphoonDataset(
            data_dir='data/processed',
            split='test',
            normalize=False,
            concat_ibtracs=False
        )
        
        logger.info(f"  ✓ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) == 0:
            logger.error("  ✗ Dataset is empty")
            return False
        
        # Load one sample
        sample = dataset[0]
        logger.info(f"  Sample keys: {list(sample.keys())}")
        logger.info(f"  past_frames: {sample['past_frames'].shape}")
        logger.info(f"  future_frames: {sample['future_frames'].shape}")
        logger.info(f"  track_past: {sample['track_past'].shape}")
        logger.info(f"  track_future: {sample['track_future'].shape}")
        
        # Check for NaN
        has_nan = any([
            torch.isnan(sample['past_frames']).any(),
            torch.isnan(sample['future_frames']).any(),
            torch.isnan(sample['track_past']).any(),
            torch.isnan(sample['track_future']).any()
        ])
        
        if has_nan:
            logger.error("  ✗ Sample contains NaN values")
            return False
        
        logger.info("  ✓ Dataset loader works correctly")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_models():
    """Check model checkpoints if they exist"""
    logger.info("\n" + "="*60)
    logger.info("4. CHECKING MODELS")
    logger.info("="*60)
    
    checkpoints_dir = Path('checkpoints')
    
    # Check autoencoder
    ae_checkpoints = list(checkpoints_dir.glob('**/autoencoder*.pth'))
    if ae_checkpoints:
        logger.info(f"  ✓ Found autoencoder checkpoint: {ae_checkpoints[0]}")
        try:
            checkpoint = torch.load(ae_checkpoints[0], map_location='cpu')
            logger.info(f"    Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                logger.info(f"    Model parameters: {len(checkpoint['model_state_dict'])} tensors")
            logger.info("    ✓ Autoencoder checkpoint valid")
        except Exception as e:
            logger.error(f"    ✗ Error loading checkpoint: {e}")
    else:
        logger.info("  ⚠️  No autoencoder checkpoint found")
    
    # Check diffusion model
    diff_checkpoints = list(checkpoints_dir.glob('**/diffusion*.pth'))
    if diff_checkpoints:
        logger.info(f"  ✓ Found diffusion checkpoint: {diff_checkpoints[0]}")
        try:
            checkpoint = torch.load(diff_checkpoints[0], map_location='cpu')
            logger.info(f"    Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                logger.info(f"    Model parameters: {len(checkpoint['model_state_dict'])} tensors")
            logger.info("    ✓ Diffusion checkpoint valid")
        except Exception as e:
            logger.error(f"    ✗ Error loading checkpoint: {e}")
    else:
        logger.info("  ⚠️  No diffusion checkpoint found")
    
    return True


def check_visualization():
    """Check visualization dependencies"""
    logger.info("\n" + "="*60)
    logger.info("5. CHECKING VISUALIZATION")
    logger.info("="*60)
    
    try:
        import matplotlib.pyplot as plt
        logger.info("  ✓ matplotlib available")
    except:
        logger.error("  ✗ matplotlib not available")
        return False
    
    try:
        import cartopy
        logger.info("  ✓ cartopy available (for map projections)")
    except:
        logger.warning("  ⚠️  cartopy not available (optional, will use simple plots)")
    
    return True


def main():
    logger.info("\n" + "="*80)
    logger.info("TYPHOON PREDICTION PIPELINE - COMPLETE CHECK")
    logger.info("="*80)
    
    results = {}
    
    # Run checks
    results['data'] = check_data_format()
    results['encoding'] = check_ibtracs_encoding()
    results['dataset'] = check_dataset()
    results['models'] = check_models()
    results['visualization'] = check_visualization()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {component.upper():20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n✓ ALL CHECKS PASSED - Pipeline is ready!")
        logger.info("\nNext steps:")
        logger.info("  1. Train autoencoder: python train_autoencoder.py")
        logger.info("  2. Train diffusion model: python train_diffusion.py")
        logger.info("  3. Generate predictions: python predict_and_visualize_trajectory.py")
    else:
        logger.error("\n✗ SOME CHECKS FAILED - Please fix issues above")
        
        # Specific guidance for coordinate swap issue
        if not results['data']:
            logger.error("\n⚠️  CRITICAL: Coordinates appear to be swapped (lon, lat) instead of (lat, lon)")
            logger.error("    You need to fix your preprocessing script to save coordinates as [latitude, longitude]")
            logger.error("    Current format shows longitude values (100-180) in position 0 and latitude values (5-45) in position 1")
    
    logger.info("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

