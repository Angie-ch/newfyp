"""
Combined Typhoon Data Preprocessing
Integrates ERA5 and IBTrACS data into training-ready format
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

from .era5_processor_v2 import ERA5Processor
from .ibtracs_processor import IBTrACSProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TyphoonPreprocessor:
    """
    Complete preprocessing pipeline for typhoon prediction
    
    Combines ERA5 atmospheric fields with IBTrACS track/intensity data
    """
    
    def __init__(
        self,
        era5_dir: str,
        ibtracs_file: str,
        output_dir: str,
        input_frames: int = 12,
        output_frames: int = 8,
        time_interval_hours: int = 6
    ):
        """
        Initialize typhoon preprocessor
        
        Args:
            era5_dir: Directory containing ERA5 data
            ibtracs_file: Path to IBTrACS CSV file
            output_dir: Directory to save processed data
            input_frames: Number of input timesteps
            output_frames: Number of output timesteps
            time_interval_hours: Time interval between frames
        """
        self.era5_processor = ERA5Processor(era5_dir)
        self.ibtracs_processor = IBTrACSProcessor(ibtracs_file)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.time_interval_hours = time_interval_hours
        self.total_frames = input_frames + output_frames
        
        # Statistics for normalization
        self.normalization_stats = None
    
    def process_typhoon_case(
        self,
        storm_id: str
    ) -> Optional[Dict]:
        """
        Process a complete typhoon case
        
        Args:
            storm_id: IBTrACS storm identifier
        
        Returns:
            Dictionary with all preprocessed data or None if processing fails
        """
        try:
            # Get storm data from IBTrACS
            storm_data = self.ibtracs_processor.get_storm_by_id(storm_id)
            
            if len(storm_data) < self.total_frames:
                logger.warning(f"Storm {storm_id} too short: {len(storm_data)} frames")
                return None
            
            # Resample to regular intervals
            storm_data = self.ibtracs_processor.get_timesteps_at_interval(
                storm_data, 
                self.time_interval_hours
            )
            
            # Extract track and intensity
            track = self.ibtracs_processor.extract_track(storm_data)
            intensity = self.ibtracs_processor.extract_intensity(storm_data, unit='mps')
            pressure = self.ibtracs_processor.extract_pressure(storm_data)
            category = self.ibtracs_processor.extract_category(storm_data)
            
            # Extract ERA5 frames
            frames = []
            timestamps = storm_data['ISO_TIME'].values[:self.total_frames]
            
            for i, timestamp in enumerate(timestamps):
                center = track[i]
                # Ensure center is float tuple
                center_tuple = (float(center[0]), float(center[1]))
                frame = self.era5_processor.load_timestep(
                    str(timestamp),
                    center=center_tuple
                )
                
                if frame is None:
                    logger.warning(f"Missing ERA5 data for {timestamp}")
                    return None
                
                frames.append(frame)
            
            frames = np.stack(frames, axis=0)  # (T, C, H, W)
            
            # Normalize frames
            normalized_frames, stats = self._normalize_frames(frames)
            
            # Split into input/output
            input_frames = normalized_frames[:self.input_frames]
            target_frames = normalized_frames[self.input_frames:]
            
            input_track = track[:self.input_frames]
            target_track = track[self.input_frames:self.total_frames]
            
            input_intensity = intensity[:self.input_frames]
            target_intensity = intensity[self.input_frames:self.total_frames]
            
            input_pressure = pressure[:self.input_frames]
            target_pressure = pressure[self.input_frames:self.total_frames]
            
            return {
                'storm_id': storm_id,
                'input_frames': input_frames.astype(np.float32),
                'target_frames': target_frames.astype(np.float32),
                'input_track': input_track.astype(np.float32),
                'target_track': target_track.astype(np.float32),
                'input_intensity': input_intensity.astype(np.float32),
                'target_intensity': target_intensity.astype(np.float32),
                'input_pressure': input_pressure.astype(np.float32),
                'target_pressure': target_pressure.astype(np.float32),
                'full_track': track[:self.total_frames].astype(np.float32),
                'timestamps': timestamps,
                'normalization_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error processing storm {storm_id}: {e}")
            return None
    
    def _normalize_frames(
        self,
        frames: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize frames using stored or computed statistics
        
        Args:
            frames: (T, C, H, W) frames to normalize
        
        Returns:
            Normalized frames and statistics
        """
        T, C, H, W = frames.shape
        
        if self.normalization_stats is None:
            # Compute statistics from this batch
            self.normalization_stats = {
                'mean': np.zeros(C),
                'std': np.zeros(C)
            }
            
            for c in range(C):
                self.normalization_stats['mean'][c] = frames[:, c].mean()
                self.normalization_stats['std'][c] = frames[:, c].std()
                
                # Avoid division by zero
                if self.normalization_stats['std'][c] < 1e-8:
                    self.normalization_stats['std'][c] = 1.0
        
        # Normalize
        normalized = np.zeros_like(frames)
        for c in range(C):
            normalized[:, c] = (frames[:, c] - self.normalization_stats['mean'][c]) / \
                               self.normalization_stats['std'][c]
        
        return normalized, self.normalization_stats
    
    def process_all_storms(
        self,
        start_date: str,
        end_date: str,
        min_intensity: float = 17.0,
        max_samples: Optional[int] = None
    ) -> List[str]:
        """
        Process all storms in a time range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_intensity: Minimum wind speed threshold
            max_samples: Maximum number of samples to process
        
        Returns:
            List of processed file paths
        """
        # Get storm IDs
        storm_ids = self.ibtracs_processor.get_storms_in_timerange(
            start_date, end_date, min_intensity
        )
        
        if max_samples:
            storm_ids = storm_ids[:max_samples]
        
        logger.info(f"Processing {len(storm_ids)} storms...")
        
        processed_files = []
        
        for storm_id in tqdm(storm_ids):
            # Process storm
            case_data = self.process_typhoon_case(storm_id)
            
            if case_data is not None:
                # Save to disk
                output_file = self.output_dir / f"{storm_id}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(case_data, f)
                
                processed_files.append(str(output_file))
        
        logger.info(f"Successfully processed {len(processed_files)} / {len(storm_ids)} storms")
        
        # Save normalization statistics
        if self.normalization_stats is not None:
            stats_file = self.output_dir / "normalization_stats.pkl"
            with open(stats_file, 'wb') as f:
                pickle.dump(self.normalization_stats, f)
            logger.info(f"Saved normalization statistics to {stats_file}")
        
        return processed_files
    
    def compute_global_statistics(
        self,
        storm_ids: List[str],
        num_samples: int = 100
    ) -> Dict:
        """
        Compute global statistics from a sample of storms
        
        Args:
            storm_ids: List of storm IDs
            num_samples: Number of samples to use for statistics
        
        Returns:
            Dictionary of statistics
        """
        logger.info(f"Computing global statistics from {num_samples} samples...")
        
        sample_ids = storm_ids[:min(num_samples, len(storm_ids))]
        
        all_frames = []
        
        for storm_id in tqdm(sample_ids):
            try:
                storm_data = self.ibtracs_processor.get_storm_by_id(storm_id)
                storm_data = self.ibtracs_processor.get_timesteps_at_interval(storm_data, 6)
                
                track = self.ibtracs_processor.extract_track(storm_data)
                
                # Load a few frames
                for i in range(min(5, len(track))):
                    center = track[i]
                    center_tuple = (float(center[0]), float(center[1]))
                    frame = self.era5_processor.load_timestep(
                        str(storm_data['ISO_TIME'].values[i]),
                        center=center_tuple
                    )
                    if frame is not None:
                        all_frames.append(frame)
            except Exception as e:
                logger.warning(f"Error loading storm {storm_id}: {e}")
                continue
        
        if not all_frames:
            logger.error("No frames loaded for statistics computation")
            return None
        
        # Compute statistics per channel across all frames
        # (frames may have different spatial dimensions)
        n_channels = all_frames[0].shape[0]
        
        all_values_per_channel = [[] for _ in range(n_channels)]
        
        for frame in all_frames:
            for c in range(n_channels):
                all_values_per_channel[c].extend(frame[c].flatten())
        
        # Compute mean and std for each channel
        stats = {
            'mean': np.array([np.mean(vals) for vals in all_values_per_channel]),
            'std': np.array([np.std(vals) for vals in all_values_per_channel])
        }
        
        # Avoid division by zero
        stats['std'] = np.maximum(stats['std'], 1e-8)
        
        self.normalization_stats = stats
        
        logger.info("Global statistics computed successfully")
        logger.info(f"  Number of channels: {n_channels}")
        logger.info(f"  Number of frames used: {len(all_frames)}")
        
        return stats

