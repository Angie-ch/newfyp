"""
Quick test to verify TyphoonDataset can load temporal split data correctly
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.datasets.typhoon_dataset import TyphoonDataset
from torch.utils.data import DataLoader

def test_dataset_loading():
    """Test loading the temporal split dataset"""
    
    print("="*80)
    print("TESTING TYPHOON DATASET LOADING")
    print("="*80)
    
    data_dir = "data/data/processed_temporal_split"
    
    # Test each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*80}")
        print(f"Testing {split.upper()} split...")
        print('='*80)
        
        try:
            # Load dataset
            dataset = TyphoonDataset(
                data_dir=data_dir,
                split=split,
                use_temporal_split=True,
                normalize=True,
                concat_ibtracs=False
            )
            
            print(f"✅ Dataset loaded successfully")
            print(f"   Number of samples: {len(dataset)}")
            
            # Test loading one sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\n📦 Sample structure:")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   {key}: {value}")
                
                # Test DataLoader
                dataloader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=0
                )
                
                print(f"\n🔄 Testing DataLoader...")
                batch = next(iter(dataloader))
                print(f"✅ DataLoader works!")
                print(f"   Batch size: {batch['past_frames'].shape[0]}")
                print(f"   Past frames batch shape: {batch['past_frames'].shape}")
                print(f"   Future frames batch shape: {batch['future_frames'].shape}")
                
            else:
                print(f"⚠️  Warning: {split} split is empty!")
                
        except Exception as e:
            print(f"❌ Error loading {split} split:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\n🎉 Dataset is ready for training!")
    return True


if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)

