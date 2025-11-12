"""
ERA5 Data Processing Module - Updated for actual file structure
Handles extraction and preprocessing of ERA5 reanalysis data
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERA5Processor:
    """
    Process ERA5 reanalysis data for typhoon prediction
    
    Variables extracted from ERA5 pressure level files:
    - z: Geopotential (m^2/s^2)
    - r: Relative humidity (%)
    - q: Specific humidity (kg/kg)
    - t: Temperature (K)
    - u: U component of wind (m/s)
    - v: V component of wind (m/s)
    - vo: Vorticity (s^-1)
    
    Total: 7 variables × 4 pressure levels = 28 channels
    """
    
    def __init__(self, data_dir: str, resolution: float = 0.25):
        """
        Initialize ERA5 processor
        
        Args:
            data_dir: Directory containing ERA5 netCDF files
            resolution: Spatial resolution in degrees (0.25 or 0.5)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        
    def extract_region(
        self,
        data: xr.Dataset,
        center: Tuple[float, float],
        size: float = 10.0
    ) -> xr.Dataset:
        """
        Extract a square region around a point
        
        Args:
            data: ERA5 dataset
            center: (latitude, longitude) of center point
            size: Size of region in degrees
        
        Returns:
            Extracted region dataset
        """
        lat, lon = center
        half_size = size / 2
        
        # Handle longitude wraparound
        lon_min = lon - half_size
        lon_max = lon + half_size
        
        # Get coordinate names (might be 'latitude' or 'lat', etc.)
        lat_name = 'latitude' if 'latitude' in data.dims else 'lat'
        lon_name = 'longitude' if 'longitude' in data.dims else 'lon'
        
        # Select region (latitude usually goes from high to low in ERA5)
        try:
            region = data.sel(
                {lat_name: slice(lat + half_size, lat - half_size),
                 lon_name: slice(lon_min, lon_max)}
            )
        except:
            # Try reverse order
            region = data.sel(
                {lat_name: slice(lat - half_size, lat + half_size),
                 lon_name: slice(lon_min, lon_max)}
            )
        
        return region
    
    def load_timestep(
        self, 
        timestamp: str,
        center: Tuple[float, float],
        time_index: int = 0
    ) -> Optional[np.ndarray]:
        """
        Load and process a single timestep
        
        Args:
            timestamp: ISO format timestamp (e.g., '2020-08-15T00:00:00')
            center: (lat, lon) center coordinates
            time_index: Which time index to use within the file (0-3 for 6-hourly data)
        
        Returns:
            Stacked array of shape (C, H, W) where C is number of channels
        """
        try:
            # Extract date and time from timestamp
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
            year = dt.year
            date_compact = dt.strftime('%Y%m%d')
            hour = dt.hour
            
            # Map hour to file time index (files have 00, 06, 12, 18 UTC)
            time_idx = hour // 6
            
            # Try multiple file naming patterns and locations
            possible_files = [
                self.data_dir / f"ERA5_{year}_26data" / f"era5_pl_{date_compact}.nc",
                self.data_dir / f"era5_pl_{date_compact}.nc",
                self.data_dir / f"era5_pressure_{date_str}.nc",
            ]
            
            pressure_file = None
            for pf in possible_files:
                if pf.exists():
                    pressure_file = pf
                    break
            
            if pressure_file is None:
                # Not an error - just missing data
                return None
                
            ds = xr.open_dataset(pressure_file)
            
            # Check if time dimension exists
            if 'valid_time' in ds.dims and len(ds.valid_time) > time_idx:
                # Select specific time
                ds_time = ds.isel(valid_time=time_idx)
            elif 'time' in ds.dims:
                ds_time = ds.isel(time=time_idx)
            else:
                ds_time = ds
            
            # Extract region around center
            region = self.extract_region(ds_time, center, size=10.0)
            
            # Stack all variables
            channels = []
            
            # Variable names in the files
            var_names = ['z', 'r', 'q', 't', 'u', 'v', 'vo']
            
            # Get pressure level coordinate name
            level_name = 'pressure_level' if 'pressure_level' in region.dims else 'level'
            
            # Stack variables for each pressure level
            if level_name in region.dims:
                levels = region[level_name].values
                for var in var_names:
                    if var in region.data_vars:
                        for level in levels:
                            try:
                                data = region[var].sel({level_name: level}).values
                                # Ensure 2D
                                if data.ndim == 0:
                                    data = np.array([[data]])
                                elif data.ndim == 1:
                                    data = data.reshape(1, -1)
                                channels.append(data)
                            except Exception as e:
                                logger.warning(f"Error extracting {var} at level {level}: {e}")
                                # Add zeros as placeholder
                                lat_size = len(region['latitude' if 'latitude' in region.dims else 'lat'])
                                lon_size = len(region['longitude' if 'longitude' in region.dims else 'lon'])
                                channels.append(np.zeros((lat_size, lon_size)))
            else:
                # No pressure levels, just stack all variables
                for var in var_names:
                    if var in region.data_vars:
                        data = region[var].values
                        if data.ndim == 0:
                            data = np.array([[data]])
                        elif data.ndim == 1:
                            data = data.reshape(1, -1)
                        channels.append(data)
            
            if len(channels) == 0:
                logger.warning(f"No variables extracted for {timestamp}")
                return None
            
            # Stack into (C, H, W)
            stacked = np.stack(channels, axis=0)
            
            ds.close()
            
            return stacked
            
        except Exception as e:
            logger.error(f"Error processing timestep {timestamp}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def normalize_frame(
        self,
        frame: np.ndarray,
        stats: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize frame using channel-wise statistics
        
        Args:
            frame: (C, H, W) frame to normalize
            stats: Pre-computed statistics or None to compute from data
        
        Returns:
            Normalized frame and statistics dictionary
        """
        if stats is None:
            # Compute statistics per channel
            C = frame.shape[0]
            stats = {
                'mean': np.zeros(C),
                'std': np.zeros(C)
            }
            
            for c in range(C):
                stats['mean'][c] = frame[c].mean()
                stats['std'][c] = frame[c].std()
                
                # Avoid division by zero
                if stats['std'][c] < 1e-8:
                    stats['std'][c] = 1.0
        
        # Normalize
        normalized = np.zeros_like(frame)
        for c in range(frame.shape[0]):
            normalized[c] = (frame[c] - stats['mean'][c]) / stats['std'][c]
        
        return normalized, stats
    
    def denormalize_frame(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Denormalize frame back to physical units
        
        Args:
            frame: (C, H, W) normalized frame
            stats: Statistics dictionary from normalization
        
        Returns:
            Denormalized frame
        """
        denormalized = np.zeros_like(frame)
        for c in range(frame.shape[0]):
            denormalized[c] = frame[c] * stats['std'][c] + stats['mean'][c]
        
        return denormalized


