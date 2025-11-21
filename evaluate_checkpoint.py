"""
Quick evaluation script to check if checkpoint meets quality targets
"""

import torch
from pathlib import Path

def evaluate_checkpoint(checkpoint_path: str):
    """
    Analyze checkpoint and provide recommendation
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    print("="*80)
    print("CHECKPOINT ANALYSIS")
    print("="*80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    epoch = ckpt.get('epoch', 'N/A')
    val_loss = ckpt.get('val_loss', 'N/A')
    config = ckpt.get('config', {})
    
    print(f"  Epoch: {epoch}")
    print(f"  Val Loss: {val_loss:.6f}")
    print(f"\n  Loss Weights:")
    era5_weight = config.get('era5_weight', 1.0)
    track_weight = config.get('track_weight', 1.0)
    intensity_weight = config.get('intensity_weight', 1.0)
    print(f"    ERA5:      {era5_weight}")
    print(f"    Track:    {track_weight}")
    print(f"    Intensity: {intensity_weight}")
    
    # Check weight configuration
    print(f"\n" + "="*80)
    print("CONFIGURATION ANALYSIS")
    print("="*80)
    
    recommended_weights = {
        'era5_weight': 1.0,
        'track_weight': 10.0,
        'intensity_weight': 5.0
    }
    
    weight_issue = False
    if track_weight != recommended_weights['track_weight']:
        print(f"\n⚠️  Track weight is {track_weight}, recommended: {recommended_weights['track_weight']}")
        weight_issue = True
    if intensity_weight != recommended_weights['intensity_weight']:
        print(f"⚠️  Intensity weight is {intensity_weight}, recommended: {recommended_weights['intensity_weight']}")
        weight_issue = True
    
    if not weight_issue:
        print(f"\n✅ Loss weights match recommendations")
    
    # Analyze total loss
    print(f"\n" + "="*80)
    print("LOSS ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Validation Loss: {val_loss:.6f}")
    
    # With equal weights (1.0 each), total = era5_loss + track_loss + intensity_loss
    # If weights are equal, we can't tell individual component performance from total
    if era5_weight == track_weight == intensity_weight == 1.0:
        print(f"\n⚠️  All weights are equal (1.0)")
        print(f"   Total loss = ERA5_loss + Track_loss + Intensity_loss")
        print(f"   Cannot determine individual component quality from total loss alone")
        print(f"   Average per-component loss ≈ {val_loss/3:.6f}")
    else:
        print(f"\n✅ Weighted loss configuration")
        print(f"   Total loss = {era5_weight}×ERA5 + {track_weight}×Track + {intensity_weight}×Intensity")
    
    # Expected targets
    print(f"\n" + "="*80)
    print("EXPECTED TARGETS (from documentation)")
    print("="*80)
    print(f"  ERA5 MSE:      < 0.01 (normalized)")
    print(f"  Track MAE:      < 0.5° (physical units)")
    print(f"  Intensity MAE: < 2.0 m/s (physical units)")
    
    # Recommendation
    print(f"\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if weight_issue:
        print(f"\n❌ RETRAIN RECOMMENDED")
        print(f"\nReason:")
        print(f"  - Loss weights are not optimal")
        print(f"  - Current: track={track_weight}, intensity={intensity_weight}")
        print(f"  - Recommended: track=10.0, intensity=5.0")
        print(f"\nImpact:")
        print(f"  - Model may not prioritize track/intensity accuracy enough")
        print(f"  - Track and intensity predictions may be less accurate")
        print(f"\nAction:")
        print(f"  - Retrain with recommended weights for better performance")
    else:
        print(f"\n✅ WEIGHTS ARE CORRECT")
        print(f"\nHowever:")
        print(f"  - Total loss of {val_loss:.6f} is reasonable")
        print(f"  - But cannot verify individual component targets without evaluation")
        print(f"\nTo fully verify quality:")
        print(f"  - Run full evaluation on validation set")
        print(f"  - Check if ERA5 MSE < 0.01, Track MAE < 0.5°, Intensity MAE < 2 m/s")
        print(f"\nFor now:")
        print(f"  - Checkpoint appears usable for next stage (diffusion training)")
        print(f"  - Monitor downstream performance")
    
    print("="*80)
    
    return {
        'epoch': epoch,
        'val_loss': val_loss,
        'weights_ok': not weight_issue,
        'recommendation': 'retrain' if weight_issue else 'evaluate'
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze checkpoint quality')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoint/joint_autoencoder/best.pth',
                       help='Path to checkpoint file')
    
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint)

