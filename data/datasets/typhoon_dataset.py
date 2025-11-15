"""
Typhoon Dataset for loading processed ERA5 + IBTrACS data
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Any
import json


class TyphoonDataset(Dataset):
    """
    Dataset for loading typhoon prediction data with ERA5 frames and IBTrACS tracks.
    
    Expected data structure:
        data_dir/
            train/
                cases/
                    *.npz files
            val/
                cases/
                    *.npz files
            test/
                cases/
                    *.npz files
    
    Each .npz file contains:
        - past_frames: (T_past, C, H, W) - ERA5 frames for past timesteps
        - future_frames: (T_future, C, H, W) - ERA5 frames for future timesteps
        - track_past: (T_past, 2) - past track positions [lat, lon]
        - track_future: (T_future, 2) - future track positions
        - intensity_past: (T_past,) - past wind intensities
        - intensity_future: (T_future,) - future wind intensities
        - pressure_past: (T_past,) - past central pressures
        - pressure_future: (T_future,) - future central pressures
        - case_id, storm_id, storm_name, year, sample_index: metadata
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        normalize: bool = True,
        concat_ibtracs: bool = False,
        use_temporal_split: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing processed data
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize ERA5 frames using global statistics
            concat_ibtracs: Whether to concatenate IBTrACS channels to ERA5 frames
            use_temporal_split: Whether to use temporal split structure (train/val/test subdirs)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        self.concat_ibtracs = concat_ibtracs
        self.use_temporal_split = use_temporal_split
        
        # Find the cases directory
        if use_temporal_split:
            # Structure: data_dir/train/cases/ or data_dir/val/cases/
            cases_dir = self.data_dir / split / 'cases'
            if not cases_dir.exists():
                # Try without 'cases' subdirectory
                cases_dir = self.data_dir / split
        else:
            # Flat structure: data_dir/cases/
            cases_dir = self.data_dir / 'cases'
            if not cases_dir.exists():
                cases_dir = self.data_dir
        
        if not cases_dir.exists():
            raise ValueError(f"Data directory not found: {cases_dir}")
        
        # Find all .npz files (exclude macOS resource fork files)
        all_files = list(cases_dir.glob('*.npz'))
        self.sample_files = sorted([
            f for f in all_files 
            if not f.name.startswith('._')  # Skip macOS resource fork files
        ])
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No .npz files found in {cases_dir}")
        
        # Load normalization statistics if needed
        self.stats = None
        if normalize:
            stats_file = self.data_dir / 'statistics.json'
            if not stats_file.exists() and use_temporal_split:
                # Try in parent directory
                stats_file = self.data_dir.parent / 'statistics.json'
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            else:
                print(f"Warning: statistics.json not found at {stats_file}. "
                      "Normalization will be skipped.")
                self.normalize = False
    
    def __len__(self) -> int:
        return len(self.sample_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - past_frames: (T_past, C, H, W) torch.Tensor
                - future_frames: (T_future, C, H, W) torch.Tensor
                - track_past: (T_past, 2) torch.Tensor
                - track_future: (T_future, 2) torch.Tensor
                - intensity_past: (T_past,) torch.Tensor
                - intensity_future: (T_future,) torch.Tensor
                - pressure_past: (T_past,) torch.Tensor (optional)
                - pressure_future: (T_future,) torch.Tensor (optional)
                - case_id: str
                - storm_id: str
                - storm_name: str (optional)
                - year: int (optional)
        """
        # Load .npz file
        sample_file = self.sample_files[idx]
        data = np.load(sample_file, allow_pickle=True)
        
        # Extract arrays
        past_frames = data['past_frames'].astype(np.float32)
        future_frames = data['future_frames'].astype(np.float32)
        track_past = data['track_past'].astype(np.float32)
        track_future = data['track_future'].astype(np.float32)
        intensity_past = data['intensity_past'].astype(np.float32)
        intensity_future = data['intensity_future'].astype(np.float32)
        
        # Normalize ERA5 frames if requested
        if self.normalize and self.stats is not None:
            mean = np.array(self.stats.get('mean', [0.0]))
            std = np.array(self.stats.get('std', [1.0]))
            
            # Ensure mean and std have correct shape for broadcasting
            if mean.ndim == 0:
                mean = mean.reshape(1, 1, 1)
            elif mean.ndim == 1:
                mean = mean.reshape(-1, 1, 1)
            
            if std.ndim == 0:
                std = std.reshape(1, 1, 1)
            elif std.ndim == 1:
                std = std.reshape(-1, 1, 1)
            
            past_frames = (past_frames - mean) / (std + 1e-8)
            future_frames = (future_frames - mean) / (std + 1e-8)
        
        # Normalize Track and Intensity if statistics are available
        if self.normalize and self.stats is not None:
            # Track normalization (lat, lon)
            track_mean = np.array(self.stats.get('track_mean', [70.0, 130.0]))  # Default: approximate center
            track_std = np.array(self.stats.get('track_std', [30.0, 30.0]))      # Default: approximate std
            
            # Ensure track_mean and track_std have shape (2,) for broadcasting
            if track_mean.ndim == 0:
                track_mean = np.array([track_mean, track_mean])
            if track_std.ndim == 0:
                track_std = np.array([track_std, track_std])
            
            # Normalize: (x - mean) / std
            track_past = (track_past - track_mean) / (track_std + 1e-8)
            track_future = (track_future - track_mean) / (track_std + 1e-8)
            
            # Intensity normalization (wind speed)
            intensity_mean = self.stats.get('intensity_mean', 10.0)  # Default: approximate mean
            intensity_std = self.stats.get('intensity_std', 10.0)    # Default: approximate std
            
            intensity_past = (intensity_past - intensity_mean) / (intensity_std + 1e-8)
            intensity_future = (intensity_future - intensity_mean) / (intensity_std + 1e-8)
        
        # Convert to torch tensors
        sample = {
            'past_frames': torch.from_numpy(past_frames),
            'future_frames': torch.from_numpy(future_frames),
            'track_past': torch.from_numpy(track_past),
            'track_future': torch.from_numpy(track_future),
            'intensity_past': torch.from_numpy(intensity_past),
            'intensity_future': torch.from_numpy(intensity_future),
        }
        
        # Add pressure if available
        if 'pressure_past' in data:
            sample['pressure_past'] = torch.from_numpy(
                data['pressure_past'].astype(np.float32)
            )
        if 'pressure_future' in data:
            sample['pressure_future'] = torch.from_numpy(
                data['pressure_future'].astype(np.float32)
            )
        
        # Add metadata
        sample['case_id'] = str(data.get('case_id', sample_file.stem))
        sample['storm_id'] = str(data.get('storm_id', ''))
        
        if 'storm_name' in data:
            sample['storm_name'] = str(data['storm_name'])
        
        # Always include 'year' field for consistent batching
        # Extract year from case_id if not in data, or use default
        if 'year' in data:
            sample['year'] = int(data['year'])
        else:
            # Try to extract year from case_id (format: "YYYYMMDD_HH")
            try:
                year_str = sample['case_id'].split('_')[0][:4]
                sample['year'] = int(year_str)
            except (ValueError, IndexError):
                sample['year'] = 2000  # Default year
        
        if 'sample_index' in data:
            sample['sample_index'] = int(data['sample_index'])
        
        return sample

