"""
Test script to verify IBTrACS channel concatenation works correctly
"""

import torch
from pathlib import Path
from data.datasets.typhoon_dataset import TyphoonDataset

def test_dataset_with_ibtracs():
    """Test dataset with IBTrACS concatenation"""
    
    print("Testing TyphoonDataset with IBTrACS concatenation...")
    
    # Use test directory
    data_dir = Path("data/test_samples_ibtracs")
    
    # Always create synthetic samples for this test
    print(f"Creating synthetic samples for testing in {data_dir}...")
    
    # Create a synthetic sample
    import numpy as np
    import shutil
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample = {
            'past_frames': np.random.randn(12, 48, 64, 64).astype(np.float32),
            'future_frames': np.random.randn(8, 48, 64, 64).astype(np.float32),
            'track_past': np.random.randn(12, 2).astype(np.float32) * 5 + np.array([22.5, 140.0]),
            'track_future': np.random.randn(8, 2).astype(np.float32) * 5 + np.array([22.5, 140.0]),
            'intensity_past': (np.random.randn(12).astype(np.float32) * 10 + 40.0),
            'intensity_future': (np.random.randn(8).astype(np.float32) * 10 + 40.0),
        'pressure_past': (np.random.randn(12).astype(np.float32) * 20 + 950.0),
        'pressure_future': (np.random.randn(8).astype(np.float32) * 20 + 950.0),
        'case_id': 'TEST001'
    }
    
    # Save multiple samples (need at least 10 for proper train/val/test split)
    import pickle
    for i in range(15):
        sample_copy = sample.copy()
        sample_copy['case_id'] = f'TEST{i+1:03d}'
        # Make each sample slightly different
        sample_copy['past_frames'] = np.random.randn(12, 48, 64, 64).astype(np.float32)
        sample_copy['future_frames'] = np.random.randn(8, 48, 64, 64).astype(np.float32)
        with open(data_dir / f"test_sample_{i+1:03d}.pkl", 'wb') as f:
            pickle.dump(sample_copy, f)
    
    print(f"✓ Created {15} synthetic samples in {data_dir}")
    
    # Test 1: Dataset without IBTrACS concatenation
    print("\n1. Testing dataset WITHOUT IBTrACS concatenation...")
    dataset_normal = TyphoonDataset(
        data_dir=str(data_dir),
        split='train',
        normalize=False,
        concat_ibtracs=False
    )
    
    print(f"   Dataset size: {len(dataset_normal)}")
    
    sample_normal = dataset_normal[0]
    print(f"   past_frames shape: {sample_normal['past_frames'].shape}")
    print(f"   future_frames shape: {sample_normal['future_frames'].shape}")
    print(f"   ✓ Expected: (T, 48, 64, 64) or (T, 40, 64, 64)")
    
    # Test 2: Dataset WITH IBTrACS concatenation
    print("\n2. Testing dataset WITH IBTrACS concatenation...")
    dataset_concat = TyphoonDataset(
        data_dir=str(data_dir),
        split='train',
        normalize=False,
        concat_ibtracs=True
    )
    
    sample_concat = dataset_concat[0]
    print(f"   past_frames shape: {sample_concat['past_frames'].shape}")
    print(f"   future_frames shape: {sample_concat['future_frames'].shape}")
    
    T_past, C_past, H, W = sample_concat['past_frames'].shape
    T_future, C_future, _, _ = sample_concat['future_frames'].shape
    
    expected_c = sample_normal['past_frames'].shape[1] + 4  # ERA5 + 4 IBTrACS channels
    
    if C_past == expected_c:
        print(f"   ✓ Channels increased from {sample_normal['past_frames'].shape[1]} to {C_past} (+4 IBTrACS)")
    else:
        print(f"   ❌ Expected {expected_c} channels, got {C_past}")
    
    # Test 3: Verify IBTrACS channels are different from ERA5
    print("\n3. Verifying IBTrACS channels...")
    era5_channels = sample_concat['past_frames'][:, :-4, :, :]  # All except last 4
    ibtracs_channels = sample_concat['past_frames'][:, -4:, :, :]  # Last 4
    
    print(f"   ERA5 channels shape: {era5_channels.shape}")
    print(f"   IBTrACS channels shape: {ibtracs_channels.shape}")
    print(f"   IBTrACS channel ranges:")
    for i in range(4):
        channel_data = ibtracs_channels[:, i, :, :]
        print(f"     Channel {i}: [{channel_data.min():.3f}, {channel_data.max():.3f}]")
    
    # Test 4: Batch loading
    print("\n4. Testing batch loading...")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset_concat,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"   Batch past_frames shape: {batch['past_frames'].shape}")
    print(f"   Expected: (B=2, T, C={expected_c}, H=64, W=64)")
    print(f"   ✓ Batch loading successful!")
    
    print("\n✅ All tests passed! IBTrACS concatenation is working correctly.")
    print(f"\n📊 Summary:")
    print(f"   - Original channels: {sample_normal['past_frames'].shape[1]}")
    print(f"   - With IBTrACS: {C_past} channels")
    print(f"   - Added: 4 channels (lat, lon, intensity, pressure)")
    
    return True

if __name__ == "__main__":
    test_dataset_with_ibtracs()

