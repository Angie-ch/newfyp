"""
Typhoon Dataset for loading processed temporal split data
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


class TyphoonDataset:
    """
    Dataset for loading typhoon data from processed temporal split structure
    
    Expected directory structure:
        data_dir/
            train/cases/*.npz
            val/cases/*.npz
            test/cases/*.npz
            normalization_stats.pkl
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        normalize: bool = True,
        use_temporal_split: bool = True,
        concat_ibtracs: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory containing train/val/test splits
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize data
            use_temporal_split: Whether to use temporal split structure
            concat_ibtracs: Whether to concatenate IBTrACS data to ERA5
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        self.use_temporal_split = use_temporal_split
        self.concat_ibtracs = concat_ibtracs
        
        # Determine data directory based on split structure
        if use_temporal_split:
            # Temporal split structure: data_dir/train/cases/, data_dir/val/cases/, etc.
            if (self.data_dir / split / 'cases').exists():
                self.cases_dir = self.data_dir / split / 'cases'
                self.stats_file = self.data_dir / 'normalization_stats.pkl'
            else:
                # Fallback: check if split directory exists
                if (self.data_dir / split).exists():
                    self.cases_dir = self.data_dir / split
                    self.stats_file = self.data_dir / 'normalization_stats.pkl'
                else:
                    # Fallback: use data_dir directly
                    self.cases_dir = self.data_dir
                    self.stats_file = self.data_dir / 'normalization_stats.pkl'
        else:
            # Flat structure: all files in data_dir
            self.cases_dir = self.data_dir
            self.stats_file = self.data_dir / 'normalization_stats.pkl'
        
        # Load case files (exclude macOS hidden files)
        self.case_files = sorted([f for f in self.cases_dir.glob('*.npz') if not f.name.startswith('._')])
        
        if len(self.case_files) == 0:
            logger.warning(f"No .npz files found in {self.cases_dir}")
        
        logger.info(f"Loaded {len(self.case_files)} samples for {split} split")
        
        # Load normalization statistics
        self.norm_stats = None
        if normalize and self.stats_file.exists():
            with open(self.stats_file, 'rb') as f:
                self.norm_stats = pickle.load(f)
            logger.info(f"Loaded normalization statistics from {self.stats_file}")
        elif normalize:
            logger.warning(f"Normalization requested but stats file not found: {self.stats_file}")
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.case_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Returns:
            Dictionary with keys:
                - 'era5': ERA5 data tensor [past_frames, channels, H, W]
                - 'ibtracs': IBTrACS data tensor [past_frames, features] (track + intensity)
                - 'era5_future': Future ERA5 data [future_frames, channels, H, W] (optional)
                - 'ibtracs_future': Future IBTrACS data [future_frames, features] (optional)
        """
        if idx >= len(self.case_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.case_files)}")
        
        # Load data file
        case_file = self.case_files[idx]
        data = np.load(case_file, allow_pickle=True)
        
        # Extract ERA5 data (past frames)
        era5_past = data['past_frames'].astype(np.float32)  # [T, C, H, W]
        
        # Extract IBTrACS data
        track_past = data['track_past'].astype(np.float32)  # [T, 2] (lat, lon)
        intensity_past = data['intensity_past'].astype(np.float32)  # [T]
        pressure_past = data.get('pressure_past', np.zeros_like(intensity_past)).astype(np.float32)  # [T]
        
        # Combine IBTrACS features: [T, 2+1+1] = [T, 4]
        # Features: lat, lon, intensity, pressure
        ibtracs_past = np.concatenate([
            track_past,  # [T, 2]
            intensity_past[:, None],  # [T, 1]
            pressure_past[:, None]  # [T, 1]
        ], axis=1)  # [T, 4]
        
        # Normalize if requested
        if self.normalize and self.norm_stats is not None:
            # Normalize ERA5
            era5_mean = self.norm_stats.get('era5_mean')
            era5_std = self.norm_stats.get('era5_std')
            if era5_mean is not None and era5_std is not None:
                # Handle different shapes: mean/std might be [C] or [C, H, W]
                if era5_mean.ndim == 1:
                    era5_past = (era5_past - era5_mean[:, None, None]) / (era5_std[:, None, None] + 1e-8)
                else:
                    era5_past = (era5_past - era5_mean) / (era5_std + 1e-8)
            
            # Normalize IBTrACS
            track_mean = self.norm_stats.get('track_mean')
            track_std = self.norm_stats.get('track_std')
            if track_mean is not None and track_std is not None:
                ibtracs_past[:, :2] = (ibtracs_past[:, :2] - track_mean) / (track_std + 1e-8)
            
            intensity_mean = self.norm_stats.get('intensity_mean')
            intensity_std = self.norm_stats.get('intensity_std')
            if intensity_mean is not None and intensity_std is not None:
                ibtracs_past[:, 2] = (ibtracs_past[:, 2] - intensity_mean) / (intensity_std + 1e-8)
            
            pressure_mean = self.norm_stats.get('pressure_mean')
            pressure_std = self.norm_stats.get('pressure_std')
            if pressure_mean is not None and pressure_std is not None:
                ibtracs_past[:, 3] = (ibtracs_past[:, 3] - pressure_mean) / (pressure_std + 1e-8)
        
        # Convert to tensors
        era5_tensor = torch.from_numpy(era5_past)  # [T, C, H, W]
        ibtracs_tensor = torch.from_numpy(ibtracs_past)  # [T, 4]
        
        # If concat_ibtracs is True, concatenate IBTrACS to ERA5 channels
        # This creates a unified input: [T, C+4, H, W]
        if self.concat_ibtracs:
            # Expand IBTrACS to spatial dimensions: [T, 4] -> [T, 4, H, W]
            T, C, H, W = era5_tensor.shape
            ibtracs_expanded = ibtracs_tensor.unsqueeze(-1).unsqueeze(-1)  # [T, 4, 1, 1]
            ibtracs_expanded = ibtracs_expanded.expand(-1, -1, H, W)  # [T, 4, H, W]
            
            # Concatenate: [T, C, H, W] + [T, 4, H, W] -> [T, C+4, H, W]
            era5_tensor = torch.cat([era5_tensor, ibtracs_expanded], dim=1)
        
        # Prepare output dictionary with keys expected by trainer
        sample = {
            'past_frames': era5_tensor,  # [T, C, H, W]
            'track_past': ibtracs_tensor[:, :2],  # [T, 2] (lat, lon)
            'intensity_past': ibtracs_tensor[:, 2],  # [T] (intensity)
        }
        
        # Add future frames if available
        if 'future_frames' in data:
            era5_future = data['future_frames'].astype(np.float32)
            
            # Normalize future ERA5
            if self.normalize and self.norm_stats is not None:
                era5_mean = self.norm_stats.get('era5_mean')
                era5_std = self.norm_stats.get('era5_std')
                if era5_mean is not None and era5_std is not None:
                    if era5_mean.ndim == 1:
                        era5_future = (era5_future - era5_mean[:, None, None]) / (era5_std[:, None, None] + 1e-8)
                    else:
                        era5_future = (era5_future - era5_mean) / (era5_std + 1e-8)
            
            sample['future_frames'] = torch.from_numpy(era5_future)
        
        if 'track_future' in data and 'intensity_future' in data:
            track_future = data['track_future'].astype(np.float32)
            intensity_future = data['intensity_future'].astype(np.float32)
            
            # Normalize future IBTrACS
            if self.normalize and self.norm_stats is not None:
                track_mean = self.norm_stats.get('track_mean')
                track_std = self.norm_stats.get('track_std')
                if track_mean is not None and track_std is not None:
                    track_future = (track_future - track_mean) / (track_std + 1e-8)
                
                intensity_mean = self.norm_stats.get('intensity_mean')
                intensity_std = self.norm_stats.get('intensity_std')
                if intensity_mean is not None and intensity_std is not None:
                    intensity_future = (intensity_future - intensity_mean) / (intensity_std + 1e-8)
            
            sample['track_future'] = torch.from_numpy(track_future)
            sample['intensity_future'] = torch.from_numpy(intensity_future)
        
        # Also keep original keys for backward compatibility
        sample['era5'] = era5_tensor
        sample['ibtracs'] = ibtracs_tensor
        
        return sample
