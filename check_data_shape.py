"""Check the shape of data in the dataset"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.datasets import TyphoonDataset

dataset = TyphoonDataset(
    data_dir="data/data/processed_temporal_split",
    split='train',
    use_temporal_split=True,
    normalize=True
)

print(f"Dataset size: {len(dataset)}")

if len(dataset) > 0:
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

