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
        
        # Find all sample files - support both .npz and .npy formats
        # .npy format: base_name_*.npy files (like LT3P)
        # .npz format: base_name.npz (legacy)
        npz_files = list(cases_dir.glob('*.npz'))
        npy_bases = set()
        
        # Find .npy files (new format like LT3P)
        for npy_file in cases_dir.glob('*_past_frames.npy'):
            base_name = npy_file.stem.replace('_past_frames', '')
            npy_bases.add(base_name)
        
        # Use .npy format if available, otherwise fall back to .npz
        if npy_bases:
            self.use_npy_format = True
            self.sample_bases = sorted([
                base for base in npy_bases
                if not base.startswith('._')  # Skip macOS resource fork files
            ])
            self.sample_files = None  # Not used for .npy format
        else:
            self.use_npy_format = False
            self.sample_files = sorted([
                f for f in npz_files 
                if not f.name.startswith('._') and not f.name.endswith('_meta.npz')
            ])
            self.sample_bases = None
        
        if not self.use_npy_format and len(self.sample_files) == 0:
            raise ValueError(f"No .npz or .npy files found in {cases_dir}")
        if self.use_npy_format and len(self.sample_bases) == 0:
            raise ValueError(f"No .npy sample files found in {cases_dir}")
        
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
        if self.use_npy_format:
            return len(self.sample_bases)
        else:
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
        # Load data - support both .npy (LT3P format) and .npz (legacy)
        if self.use_npy_format:
            # Load .npy format (like LT3P) - faster loading
            base_name = self.sample_bases[idx]
            cases_dir = self.data_dir / self.split / 'cases' if self.use_temporal_split else self.data_dir / 'cases'
            
            # Load each component from separate .npy files
            past_frames = np.load(cases_dir / f"{base_name}_past_frames.npy").astype(np.float32)
            future_frames = np.load(cases_dir / f"{base_name}_future_frames.npy").astype(np.float32)
            track_past = np.load(cases_dir / f"{base_name}_track_past.npy").astype(np.float32)
            track_future = np.load(cases_dir / f"{base_name}_track_future.npy").astype(np.float32)
            intensity_past = np.load(cases_dir / f"{base_name}_intensity_past.npy").astype(np.float32)
            intensity_future = np.load(cases_dir / f"{base_name}_intensity_future.npy").astype(np.float32)
            
            # Load metadata
            metadata_file = cases_dir / f"{base_name}_meta.npz"
            if metadata_file.exists():
                metadata = np.load(metadata_file, allow_pickle=True)
                case_id = str(metadata.get('case_id', base_name))
                storm_id = str(metadata.get('storm_id', ''))
                storm_name = str(metadata.get('storm_name', '')) if 'storm_name' in metadata else None
                year = int(metadata.get('year', 2000))
                sample_index = int(metadata.get('window_index', 0))
            else:
                # Fallback if metadata file doesn't exist
                case_id = base_name
                storm_id = ''
                storm_name = None
                year = 2000
                sample_index = 0
            
            # Load pressure if available
            pressure_past = None
            pressure_future = None
            pressure_past_file = cases_dir / f"{base_name}_pressure_past.npy"
            pressure_future_file = cases_dir / f"{base_name}_pressure_future.npy"
            if pressure_past_file.exists():
                pressure_past = np.load(pressure_past_file).astype(np.float32)
            if pressure_future_file.exists():
                pressure_future = np.load(pressure_future_file).astype(np.float32)
        else:
            # Load legacy .npz format
            sample_file = self.sample_files[idx]
            data = np.load(sample_file, allow_pickle=True)
            
            # Extract arrays
            past_frames = data['past_frames'].astype(np.float32)
            future_frames = data['future_frames'].astype(np.float32)
            track_past = data['track_past'].astype(np.float32)
            track_future = data['track_future'].astype(np.float32)
            intensity_past = data['intensity_past'].astype(np.float32)
            intensity_future = data['intensity_future'].astype(np.float32)
            
            # Extract metadata
            case_id = str(data.get('case_id', sample_file.stem))
            storm_id = str(data.get('storm_id', ''))
            storm_name = str(data['storm_name']) if 'storm_name' in data else None
            year = int(data['year']) if 'year' in data else 2000
            sample_index = int(data['sample_index']) if 'sample_index' in data else 0
            
            # Load pressure if available
            pressure_past = None
            pressure_future = None
            if 'pressure_past' in data:
                pressure_past = data['pressure_past'].astype(np.float32)
            if 'pressure_future' in data:
                pressure_future = data['pressure_future'].astype(np.float32)
        
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
        
        # Add pressure if available (already loaded above for both formats)
        if pressure_past is not None:
            sample['pressure_past'] = torch.from_numpy(pressure_past)
        if pressure_future is not None:
            sample['pressure_future'] = torch.from_numpy(pressure_future)
        
        # Add metadata (already extracted above for both formats)
        sample['case_id'] = case_id
        sample['storm_id'] = storm_id
        
        if storm_name:
            sample['storm_name'] = storm_name
        
        sample['year'] = year
        sample['sample_index'] = sample_index
        
        return sample

