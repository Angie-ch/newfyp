"""
IBTrACS Data Processing Module
Handles extraction and preprocessing of typhoon track/intensity data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IBTrACSProcessor:
    """
    Process IBTrACS (International Best Track Archive for Climate Stewardship) data
    
    Extracts:
    - Typhoon center coordinates (lat, lon)
    - Maximum sustained wind speed
    - Minimum central pressure
    - Storm category
    - Radius of maximum winds (if available)
    """
    
    # Storm intensity categories
    CATEGORIES = {
        'TD': 'Tropical Depression',
        'TS': 'Tropical Storm',
        'TY': 'Typhoon',
        'STY': 'Super Typhoon'
    }
    
    def __init__(self, data_file: str, basin: str = 'WP'):
        """
        Initialize IBTrACS processor
        
        Args:
            data_file: Path to IBTrACS CSV file
            basin: Ocean basin code (WP = Western Pacific)
        """
        self.data_file = Path(data_file)
        self.basin = basin
        self.data = None
        
        if self.data_file.exists():
            self.load_data()
        else:
            logger.warning(f"IBTrACS file not found: {data_file}")
    
    def load_data(self):
        """Load IBTrACS CSV data"""
        logger.info(f"Loading IBTrACS data from {self.data_file}")
        self.data = pd.read_csv(self.data_file, low_memory=False)
        
        # Filter by basin
        if self.basin:
            self.data = self.data[self.data['BASIN'] == self.basin]
        
        logger.info(f"Loaded {len(self.data)} records from {self.basin} basin")
    
    def get_storm_by_id(self, storm_id: str) -> pd.DataFrame:
        """
        Get all records for a specific storm
        
        Args:
            storm_id: IBTrACS storm identifier
        
        Returns:
            DataFrame with storm track
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        storm_data = self.data[self.data['SID'] == storm_id].copy()
        storm_data = storm_data.sort_values('ISO_TIME')
        
        return storm_data
    
    def get_storms_in_timerange(
        self,
        start_date: str,
        end_date: str,
        min_intensity: float = 17.0  # m/s (≈ Tropical Storm)
    ) -> List[str]:
        """
        Get list of storm IDs in a time range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_intensity: Minimum wind speed threshold
        
        Returns:
            List of storm IDs
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Convert to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Filter by time range
        mask = (pd.to_datetime(self.data['ISO_TIME']) >= start) & \
               (pd.to_datetime(self.data['ISO_TIME']) <= end)
        
        filtered = self.data[mask]
        
        # Filter by minimum intensity
        if min_intensity > 0:
            # Use WMO wind speed if available
            if 'WMO_WIND' in filtered.columns:
                # Convert to numeric, handling missing/invalid values
                wind_numeric = pd.to_numeric(filtered['WMO_WIND'], errors='coerce')
                filtered = filtered[wind_numeric >= min_intensity * 1.944]  # Convert m/s to knots
        
        # Get unique storm IDs
        storm_ids = filtered['SID'].unique().tolist()
        
        logger.info(f"Found {len(storm_ids)} storms between {start_date} and {end_date}")
        
        return storm_ids
    
    def extract_track(self, storm_data: pd.DataFrame) -> np.ndarray:
        """
        Extract center coordinates from storm data
        
        Args:
            storm_data: DataFrame for a single storm
        
        Returns:
            Array of shape (N, 2) with (lat, lon) coordinates
        """
        lats = storm_data['LAT'].values
        lons = storm_data['LON'].values
        
        # Handle missing values
        valid_mask = ~(pd.isna(lats) | pd.isna(lons))
        
        track = np.stack([lats[valid_mask], lons[valid_mask]], axis=1)
        
        return track
    
    def extract_intensity(
        self,
        storm_data: pd.DataFrame,
        unit: str = 'mps'  # 'mps' or 'knots'
    ) -> np.ndarray:
        """
        Extract maximum sustained wind speed
        
        Args:
            storm_data: DataFrame for a single storm
            unit: Output unit ('mps' for m/s, 'knots' for nautical miles/hour)
        
        Returns:
            Array of shape (N,) with wind speeds
        """
        # Prefer WMO wind speed
        if 'WMO_WIND' in storm_data.columns:
            wind = pd.to_numeric(storm_data['WMO_WIND'], errors='coerce').values
        elif 'USA_WIND' in storm_data.columns:
            wind = pd.to_numeric(storm_data['USA_WIND'], errors='coerce').values
        else:
            logger.warning("No wind speed data found")
            return np.zeros(len(storm_data))
        
        # Handle missing values
        wind = pd.Series(wind).ffill().bfill().values
        
        # Convert units if needed
        if unit == 'mps':
            wind = wind * 0.514444  # knots to m/s
        
        return wind
    
    def extract_pressure(self, storm_data: pd.DataFrame) -> np.ndarray:
        """
        Extract minimum central pressure
        
        Args:
            storm_data: DataFrame for a single storm
        
        Returns:
            Array of shape (N,) with pressures in hPa
        """
        if 'WMO_PRES' in storm_data.columns:
            pressure = pd.to_numeric(storm_data['WMO_PRES'], errors='coerce').values
        elif 'USA_PRES' in storm_data.columns:
            pressure = pd.to_numeric(storm_data['USA_PRES'], errors='coerce').values
        else:
            logger.warning("No pressure data found")
            return np.zeros(len(storm_data))
        
        # Handle missing values
        pressure = pd.Series(pressure).ffill().bfill().values
        
        return pressure
    
    def extract_category(self, storm_data: pd.DataFrame) -> np.ndarray:
        """
        Extract storm category
        
        Args:
            storm_data: DataFrame for a single storm
        
        Returns:
            Array of shape (N,) with category codes (0-4)
        """
        # Map categories to numbers
        category_map = {
            'TD': 0,    # Tropical Depression
            'TS': 1,    # Tropical Storm
            'TY': 2,    # Typhoon
            'STY': 3,   # Super Typhoon
        }
        
        if 'USA_SSHS' in storm_data.columns:
            categories = storm_data['USA_SSHS'].values
        else:
            # Infer from wind speed
            wind = self.extract_intensity(storm_data, unit='mps')
            categories = []
            for w in wind:
                if w < 17.0:
                    categories.append('TD')
                elif w < 33.0:
                    categories.append('TS')
                elif w < 51.0:
                    categories.append('TY')
                else:
                    categories.append('STY')
            categories = np.array(categories)
        
        # Convert to numbers
        numeric_categories = np.array([category_map.get(c, 0) for c in categories])
        
        return numeric_categories
    
    def get_timesteps_at_interval(
        self,
        storm_data: pd.DataFrame,
        interval_hours: int = 6
    ) -> pd.DataFrame:
        """
        Resample storm data to regular time intervals
        
        Args:
            storm_data: DataFrame for a single storm
            interval_hours: Time interval in hours
        
        Returns:
            Resampled DataFrame
        """
        # Convert to datetime
        storm_data['ISO_TIME'] = pd.to_datetime(storm_data['ISO_TIME'])
        storm_data = storm_data.set_index('ISO_TIME')
        
        # Resample to regular intervals
        resampled = storm_data.resample(f'{interval_hours}H').first()
        
        # Interpolate missing values
        resampled = resampled.interpolate(method='linear')
        
        return resampled.reset_index()
    
    def split_sequence(
        self,
        track: np.ndarray,
        intensity: np.ndarray,
        input_len: int = 12,
        output_len: int = 8
    ) -> List[Dict]:
        """
        Split storm track into input/output sequences
        
        Args:
            track: (N, 2) array of coordinates
            intensity: (N,) array of wind speeds
            input_len: Number of input timesteps
            output_len: Number of output timesteps
        
        Returns:
            List of sequences, each containing input/output splits
        """
        sequences = []
        total_len = input_len + output_len
        
        for i in range(len(track) - total_len + 1):
            sequence = {
                'input_track': track[i:i+input_len],
                'output_track': track[i+input_len:i+total_len],
                'input_intensity': intensity[i:i+input_len],
                'output_intensity': intensity[i+input_len:i+total_len],
            }
            sequences.append(sequence)
        
        return sequences

