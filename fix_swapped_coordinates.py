"""
Fix Swapped Coordinates in Preprocessed Data

This script fixes the coordinate order from [lon, lat] to [lat, lon]
"""

import numpy as np
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_coordinates(data_dir: str, backup: bool = True):
    """
    Fix swapped coordinates in all .npz files
    
    Args:
        data_dir: Directory containing the case files
        backup: Whether to create backups before modifying
    """
    cases_dir = Path(data_dir) / 'cases'
    
    if not cases_dir.exists():
        logger.error(f"Directory not found: {cases_dir}")
        return
    
    # Get valid samples (exclude macOS temp files)
    samples = [f for f in cases_dir.glob('*.npz') if not f.name.startswith('._')]
    
    logger.info(f"Found {len(samples)} files to fix")
    
    # Create backup directory if requested
    if backup:
        backup_dir = cases_dir / 'backup_original'
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"Backups will be saved to: {backup_dir}")
    
    fixed_count = 0
    
    for sample_file in samples:
        try:
            # Load data
            data = np.load(sample_file, allow_pickle=True)
            
            # Check if coordinates need fixing
            track_past = data['track_past']
            track_future = data['track_future']
            
            # Check if swapped (latitude > 100 indicates it's actually longitude)
            if track_past[0, 0] > 100:
                logger.info(f"Fixing: {sample_file.name}")
                
                # Create backup
                if backup:
                    backup_path = backup_dir / sample_file.name
                    shutil.copy2(sample_file, backup_path)
                
                # Swap coordinates: [lon, lat] → [lat, lon]
                track_past_fixed = np.stack([track_past[:, 1], track_past[:, 0]], axis=1)
                track_future_fixed = np.stack([track_future[:, 1], track_future[:, 0]], axis=1)
                
                logger.info(f"  Before: lat=[{track_past[0,0]:.1f}, {track_past[-1,0]:.1f}], "
                          f"lon=[{track_past[0,1]:.1f}, {track_past[-1,1]:.1f}]")
                logger.info(f"  After:  lat=[{track_past_fixed[0,0]:.1f}, {track_past_fixed[-1,0]:.1f}], "
                          f"lon=[{track_past_fixed[0,1]:.1f}, {track_past_fixed[-1,1]:.1f}]")
                
                # Save fixed data
                np.savez(
                    sample_file,
                    past_frames=data['past_frames'],
                    future_frames=data['future_frames'],
                    track_past=track_past_fixed,
                    track_future=track_future_fixed,
                    intensity_past=data['intensity_past'],
                    intensity_future=data['intensity_future'],
                    case_id=data['case_id']
                )
                
                fixed_count += 1
            else:
                # Coordinates look correct
                logger.debug(f"Skipping {sample_file.name} (already correct)")
                
        except Exception as e:
            logger.error(f"Error processing {sample_file.name}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Fixed {fixed_count} / {len(samples)} files")
    if backup:
        logger.info(f"✓ Original files backed up to: {backup_dir}")
    logger.info(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix swapped coordinates in preprocessed data')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backups (not recommended)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FIXING SWAPPED COORDINATES")
    logger.info("="*60)
    
    fix_coordinates(args.data_dir, backup=not args.no_backup)
    
    logger.info("\n✓ Done! Re-run check_complete_pipeline.py to verify.")


if __name__ == '__main__':
    main()















