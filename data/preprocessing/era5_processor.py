"""
ERA5 Data Processing Module
Handles extraction and preprocessing of ERA5 reanalysis data
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERA5Processor:
    """
    Process ERA5 reanalysis data for typhoon prediction
    
    Variables extracted:
    - Pressure levels (1000, 925, 850, 700, 500, 300 hPa):
        - u-wind, v-wind, temperature, geopotential height, relative humidity
    - Surface: sea level pressure
    - Single level: sea surface temperature
    """
    
    PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]  # hPa
    
    VARIABLES = {
        'u': 'u_component_of_wind',           # m/s
        'v': 'v_component_of_wind',           # m/s
        't': 'temperature',                    # K
        'z': 'geopotential',                   # m^2/s^2
        'r': 'relative_humidity',              # %
    }
    
    SURFACE_VARS = {
        'msl': 'mean_sea_level_pressure',     # Pa
        'sst': 'sea_surface_temperature',     # K
    }
    
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
        size: float = 20.0
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
        
        region = data.sel(
            latitude=slice(lat + half_size, lat - half_size),  # N to S
            longitude=slice(lon_min, lon_max)
        )
        
        return region
    
    def load_timestep(
        self, 
        timestamp: str,
        center: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """
        Load and process a single timestep
        
        Args:
            timestamp: ISO format timestamp (e.g., '2020-08-15T00:00:00')
            center: (lat, lon) center coordinates
        
        Returns:
            Stacked array of shape (C, H, W) where C is number of channels
        """
        try:
            # Extract date from timestamp
            date_str = timestamp[:10]  # YYYY-MM-DD
            year = date_str[:4]
            date_compact = date_str.replace('-', '')  # YYYYMMDD
            
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
                logger.warning(f"No ERA5 file found for {date_str}")
                return None
                
            ds_pressure = xr.open_dataset(pressure_file)
            region_pressure = self.extract_region(ds_pressure, center)
            
            # Stack all variables
            channels = []
            
            # Pressure level variables (6 levels × 5 variables = 30 channels)
            for var_short, var_long in self.VARIABLES.items():
                if var_long in region_pressure:
                    for level in self.PRESSURE_LEVELS:
                        data = region_pressure[var_long].sel(level=level).values
                        channels.append(data)
            
            # Surface variables (2 channels)
            for var_short, var_long in self.SURFACE_VARS.items():
                if var_long in region_surface:
                    data = region_surface[var_long].values
                    channels.append(data)
            
            # Stack into (C, H, W)
            stacked = np.stack(channels, axis=0)
            
            return stacked
            
        except Exception as e:
            logger.error(f"Error processing timestep {timestamp}: {e}")
            return None
    
    def compute_derived_variables(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute derived meteorological variables
        
        Args:
            frames: (T, C, H, W) atmospheric state
        
        Returns:
            Dictionary of derived variables
        """
        # Extract u and v winds (first 12 channels: 6 levels × 2 components)
        u = frames[:, 0:6, :, :]   # (T, 6, H, W)
        v = frames[:, 6:12, :, :]  # (T, 6, H, W)
        
        # Compute vorticity: ∂v/∂x - ∂u/∂y
        dv_dx = self._gradient_x(v)
        du_dy = self._gradient_y(u)
        vorticity = dv_dx - du_dy
        
        # Compute divergence: ∂u/∂x + ∂v/∂y
        du_dx = self._gradient_x(u)
        dv_dy = self._gradient_y(v)
        divergence = du_dx + dv_dy
        
        # Compute wind speed
        wind_speed = np.sqrt(u**2 + v**2)
        
        return {
            'vorticity': vorticity,
            'divergence': divergence,
            'wind_speed': wind_speed
        }
    
    def _gradient_x(self, field: np.ndarray) -> np.ndarray:
        """Compute ∂/∂x using central differences"""
        grad = np.zeros_like(field)
        grad[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2 * self.resolution)
        # Forward/backward difference at boundaries
        grad[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / self.resolution
        grad[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / self.resolution
        return grad
    
    def _gradient_y(self, field: np.ndarray) -> np.ndarray:
        """Compute ∂/∂y using central differences"""
        grad = np.zeros_like(field)
        grad[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2 * self.resolution)
        # Forward/backward difference at boundaries
        grad[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / self.resolution
        grad[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / self.resolution
        return grad
    
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

