"""
Adapt our typhoon data (72 samples) for Video Diffusion Model training
Converts from our format to forecast-video-diffmodels format
"""
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm

print("=" * 80)
print("ADAPTING DATA FOR VIDEO DIFFUSION MODEL")
print("=" * 80)

# Configuration
SOURCE_DIR = Path("D:/typhoon_data_2018_2021_full")
TARGET_DIR = Path("forecast-video-diffmodels/dataloader/64_FC")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

class VideoDataLoader:
    """
    Dataloader compatible with forecast-video-diffmodels
    Stores video sequences for diffusion model training
    """
    def __init__(self, mode="fc", o_size=64, n_size=128):
        self.mode = mode
        self.o_size = o_size
        self.n_size = n_size
        self.data = []
    
    def add_sequence(self, frames, condition):
        """
        Add a video sequence
        Args:
            frames: (T, C, H, W) - future frames to predict
            condition: (T_cond, C, H, W) - conditioning frames (past)
        """
        self.data.append({
            'frames': frames,  # Future frames
            'condition': condition  # Past frames as condition
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def normalize_frames(frames):
    """Normalize frames to [-1, 1] range for diffusion model"""
    # frames shape: (T, C, H, W)
    min_val = frames.min()
    max_val = frames.max()
    if max_val - min_val > 0:
        normalized = 2 * (frames - min_val) / (max_val - min_val) - 1
    else:
        normalized = frames
    return normalized.astype(np.float32)

def process_split(split_name):
    """Process train/val/test split"""
    split_dir = SOURCE_DIR / split_name / 'cases'
    samples = sorted(list(split_dir.glob('*.npz')))
    
    if len(samples) == 0:
        print(f"  [WARNING] No samples found in {split_name}")
        return None
    
    print(f"\n[{split_name.upper()}] Processing {len(samples)} samples...")
    
    dataloader = VideoDataLoader(mode="fc", o_size=64, n_size=128)
    
    for sample_file in tqdm(samples):
        data = np.load(sample_file)
        
        # Extract data
        past_frames = data['past_frames']    # (8, 24, 64, 64)
        future_frames = data['future_frames'] # (12, 24, 64, 64)
        
        # Normalize to [-1, 1]
        past_normalized = normalize_frames(past_frames)
        future_normalized = normalize_frames(future_frames)
        
        # Convert to torch tensors
        past_tensor = torch.from_numpy(past_normalized).float()
        future_tensor = torch.from_numpy(future_normalized).float()
        
        # Add to dataloader
        # In video diffusion: we predict future frames conditioned on past frames
        dataloader.add_sequence(
            frames=future_tensor,      # Target: predict these
            condition=past_tensor       # Condition: use these as input
        )
    
    # Save dataloader
    output_file = TARGET_DIR / f"{split_name}_dataloader.dat"
    with open(output_file, 'wb') as f:
        pickle.dump(dataloader, f)
    
    print(f"  [OK] Saved {len(dataloader)} sequences to {output_file}")
    return dataloader

# Process all splits
print("\nProcessing splits...")
train_loader = process_split('train')
val_loader = process_split('val')
test_loader = process_split('test')

# Create summary
print("\n" + "=" * 80)
print("DATA ADAPTATION SUMMARY")
print("=" * 80)
print(f"Training samples:   {len(train_loader) if train_loader else 0}")
print(f"Validation samples: {len(val_loader) if val_loader else 0}")
print(f"Test samples:       {len(test_loader) if test_loader else 0}")
print(f"Total:              {sum([len(x) for x in [train_loader, val_loader, test_loader] if x])}")

print("\nData format:")
print("  Past frames (condition): (8, 24, 64, 64)")
print("  Future frames (target):  (12, 24, 64, 64)")
print("  Normalization: [-1, 1]")
print("  Device: CPU (will move to GPU during training)")

print("\n" + "=" * 80)
print("ADAPTATION COMPLETE!")
print("=" * 80)
print(f"\nData saved to: {TARGET_DIR}")
print("\nNext steps:")
print("1. cd forecast-video-diffmodels/imagen")
print("2. pip install -r requirements.txt")
print("3. Run training script with our adapted data")
print("=" * 80)

# Create metadata file
metadata = {
    'num_train': len(train_loader) if train_loader else 0,
    'num_val': len(val_loader) if val_loader else 0,
    'num_test': len(test_loader) if test_loader else 0,
    'input_shape': (8, 24, 64, 64),
    'output_shape': (12, 24, 64, 64),
    'normalization': 'min-max to [-1, 1]',
    'source': 'typhoon_data_2018_2021_full',
}

with open(TARGET_DIR / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\nMetadata saved to: {TARGET_DIR / 'metadata.pkl'}")

