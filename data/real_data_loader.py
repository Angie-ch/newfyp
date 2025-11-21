"""
Real typhoon data loader using IBTrACS WP (Western Pacific) and ERA5 reanalysis data

Data Sources:
- IBTrACS WP: Western Pacific typhoon tracks and intensity
- ERA5: ECMWF reanalysis meteorological data (wind, pressure, temperature, humidity, etc.)
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import torch
import xarray as xr
import time
import contextlib
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
    warnings.warn("cdsapi not installed. ERA5 download will not work. Install with: pip install cdsapi")


class IBTrACSLoader:
    """
    Load real typhoon data from IBTrACS database
    IBTrACS: International Best Track Archive for Climate Stewardship
    """
    
    def __init__(self, data_dir: str = "data/raw", dataset: str = "all"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = dataset.lower()
        dataset_urls = {
            "all": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv",
            "last3years": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.last3years.list.v04r00.csv"
        }
        if dataset not in dataset_urls:
            raise ValueError(f"Unsupported IBTrACS dataset '{dataset}'. Use 'all' or 'last3years'.")
        
        self.dataset = dataset
        self.ibtracs_url = dataset_urls[dataset]
        self.cache_file = self.data_dir / f"ibtracs_wp_{dataset}.csv"
        
    def download_ibtracs(self, force_download: bool = False):
        """Download IBTrACS data for Western Pacific"""
        if self.cache_file.exists() and not force_download:
            print(f"[OK] Using cached IBTrACS data: {self.cache_file}")
            return
        
        print(f"Downloading IBTrACS Western Pacific data ({self.dataset})...")
        print(f"URL: {self.ibtracs_url}")
        
        try:
            response = requests.get(self.ibtracs_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(self.cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[OK] Downloaded IBTrACS data to {self.cache_file}")
        except Exception as e:
            print(f"[ERROR] Error downloading IBTrACS data: {e}")
            print("Tip: You can manually download from:")
            print(f"    {self.ibtracs_url}")
            print(f"    and save to {self.cache_file}")
            raise
            
    def load_ibtracs(self) -> pd.DataFrame:
        """Load and parse IBTrACS CSV data"""
        if not self.cache_file.exists():
            self.download_ibtracs()
        
        print("Loading IBTrACS data...")
        
        # Skip the first row which contains units
        df = pd.read_csv(self.cache_file, skiprows=[1], low_memory=False)
        
        print(f"[OK] Loaded {len(df)} records")
        print(f"  Unique storms: {df['SID'].nunique()}")
        
        return df
        
    def filter_typhoons(self, 
                       df: pd.DataFrame,
                       start_year: int = 2015,
                       end_year: int = 2023,
                       min_wind_speed: float = 17.0,  # m/s (~34 knots, Tropical Storm)
                       min_duration_hours: int = 24) -> List[str]:
        """
        Filter for strong typhoons in the Western Pacific basin
        
        Returns:
            List of storm IDs (SIDs)
        """
        print(f"\nFiltering WP typhoons from {start_year}-{end_year}...")
        
        # Filter for Western Pacific basin only
        if 'BASIN' in df.columns:
            df = df[df['BASIN'] == 'WP'].copy()
            print(f"  Filtered to {len(df)} WP basin records")
        
        # Convert ISO_TIME to datetime
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df['YEAR'] = df['ISO_TIME'].dt.year
        
        # Filter by year
        df_filtered = df[(df['YEAR'] >= start_year) & (df['YEAR'] <= end_year)].copy()
        
        # Get wind speed (USA_WIND is in knots, convert to m/s)
        df_filtered['WIND_MS'] = pd.to_numeric(df_filtered['USA_WIND'], errors='coerce') * 0.514444
        
        # Group by storm
        storm_stats = df_filtered.groupby('SID').agg({
            'WIND_MS': 'max',
            'ISO_TIME': ['min', 'max'],
            'SID': 'count'
        })
        
        storm_stats.columns = ['max_wind', 'start_time', 'end_time', 'n_records']
        
        # Calculate duration
        storm_stats['duration_hours'] = (
            (storm_stats['end_time'] - storm_stats['start_time']).dt.total_seconds() / 3600
        )
        
        # Filter
        strong_storms = storm_stats[
            (storm_stats['max_wind'] >= min_wind_speed) &
            (storm_stats['duration_hours'] >= min_duration_hours) &
            (storm_stats['n_records'] >= 8)  # At least 8 observations
        ]
        
        storm_ids = strong_storms.index.tolist()
        
        print(f"[OK] Found {len(storm_ids)} strong typhoons:")
        print(f"  Max wind >= {min_wind_speed} m/s")
        print(f"  Duration >= {min_duration_hours} hours")
        print(f"  Records per storm: {strong_storms['n_records'].mean():.1f} (avg)")
        
        return storm_ids
        
    def get_storm_data(self, df: pd.DataFrame, storm_id: str) -> Dict:
        """
        Extract time series data for a specific storm
        
        Returns:
            Dictionary with:
                - times: datetime array
                - lats: latitude array
                - lons: longitude array
                - winds: wind speed array (m/s)
                - pressures: pressure array (hPa)
        """
        storm_df = df[df['SID'] == storm_id].copy()
        storm_df = storm_df.sort_values('ISO_TIME')
        
        # Extract data
        times = pd.to_datetime(storm_df['ISO_TIME']).values
        lats = pd.to_numeric(storm_df['LAT'], errors='coerce').values
        lons = pd.to_numeric(storm_df['LON'], errors='coerce').values
        
        # Wind speed (convert knots to m/s)
        winds = pd.to_numeric(storm_df['USA_WIND'], errors='coerce').values * 0.514444
        
        # Pressure
        pressures = pd.to_numeric(storm_df['USA_PRES'], errors='coerce').values
        
        # Remove NaN entries
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        
        return {
            'storm_id': storm_id,
            'times': times[valid_mask],
            'lats': lats[valid_mask],
            'lons': lons[valid_mask],
            'winds': winds[valid_mask],
            'pressures': pressures[valid_mask],
            'name': storm_df['NAME'].iloc[0] if 'NAME' in storm_df.columns else 'UNKNOWN'
        }
        
    def create_training_sample(self,
                              storm_data: Dict,
                              past_timesteps: int = 12,
                              future_timesteps: int = 8,
                              image_size: Tuple[int, int] = (64, 64),  # Default 64x64 for better spatial resolution
                              n_channels: int = 48,  # ERA5 channels
                              era5_dataset: Optional[xr.Dataset] = None,
                              era5_loader: Optional['ERA5Loader'] = None,
                              use_era5: bool = True) -> Optional[Dict]:
        """
        Create a training sample from storm data
        
        REQUIRES real ERA5 data. If ERA5 data is not available, returns None.
        Never uses synthetic data.
        
        Args:
            storm_data: Storm data from IBTrACS
            past_timesteps: Number of past timesteps
            future_timesteps: Number of future timesteps
            image_size: Size of image frames (H, W)
            n_channels: Number of channels (48 for ERA5)
            era5_dataset: Pre-loaded ERA5 dataset (required if use_era5=True)
            era5_loader: ERA5Loader instance (required if use_era5=True)
            use_era5: Whether to use ERA5 data (must be True with valid ERA5 data)
        
        Returns:
            Dictionary with training data or None if not enough timesteps or no ERA5 data
        """
        n_total = len(storm_data['times'])
        
        if n_total < past_timesteps + future_timesteps:
            return None
        
        # Select a random starting point
        max_start = n_total - past_timesteps - future_timesteps
        start_idx = np.random.randint(0, max_start + 1)
        
        past_end = start_idx + past_timesteps
        future_end = past_end + future_timesteps
        
        # Extract track data
        past_lons = storm_data['lons'][start_idx:past_end]
        past_lats = storm_data['lats'][start_idx:past_end]
        future_lons = storm_data['lons'][past_end:future_end]
        future_lats = storm_data['lats'][past_end:future_end]
        
        # Extract intensity data
        past_winds = storm_data['winds'][start_idx:past_end]
        future_winds = storm_data['winds'][past_end:future_end]
        
        past_pressures = storm_data['pressures'][start_idx:past_end]
        future_pressures = storm_data['pressures'][past_end:future_end]
        
        # Extract times
        past_times = storm_data['times'][start_idx:past_end]
        future_times = storm_data['times'][past_end:future_end]
        
        # Get meteorological frames - ONLY use real ERA5 data, never synthetic
        if use_era5 and era5_dataset is not None and era5_loader is not None:
            # Use real ERA5 data
            try:
                # Use 64x64 for better spatial resolution
                crop_size = image_size[0] if image_size is not None else 64
                
                past_frames = era5_loader.extract_frames_at_times(
                    era5_dataset, past_lons, past_lats, past_times,
                    crop_size=crop_size,
                    load_per_timestep=True
                )
                future_frames = era5_loader.extract_frames_at_times(
                    era5_dataset, future_lons, future_lats, future_times,
                    crop_size=crop_size,
                    load_per_timestep=True
                )
                
                # Validate that frames contain real data (not all zeros or NaN)
                if past_frames is None:
                    print(f"Warning: ERA5 extraction returned None for past_frames, skipping sample for {storm_data.get('storm_id', 'unknown')}")
                    return None
                if future_frames is None:
                    print(f"Warning: ERA5 extraction returned None for future_frames, skipping sample for {storm_data.get('storm_id', 'unknown')}")
                    return None
                
                # Check for zeros or NaN
                if np.all(past_frames == 0) or np.all(np.isnan(past_frames)):
                    print(f"Warning: ERA5 extraction returned all zeros/NaN for past_frames, skipping sample for {storm_data.get('storm_id', 'unknown')}")
                    return None
                if np.all(future_frames == 0) or np.all(np.isnan(future_frames)):
                    print(f"Warning: ERA5 extraction returned all zeros/NaN for future_frames, skipping sample for {storm_data.get('storm_id', 'unknown')}")
                    return None
                
                # Check that we have some non-zero, non-NaN values
                if np.count_nonzero(~np.isnan(past_frames)) == 0:
                    print(f"Warning: ERA5 extraction returned no valid data for past_frames, skipping sample for {storm_data.get('storm_id', 'unknown')}")
                    return None
                if np.count_nonzero(~np.isnan(future_frames)) == 0:
                    print(f"Warning: ERA5 extraction returned no valid data for future_frames, skipping sample for {storm_data.get('storm_id', 'unknown')}")
                    return None
                    
            except Exception as e:
                # ERA5 extraction failed - return None instead of using synthetic data
                print(f"Warning: ERA5 extraction failed for {storm_data.get('storm_id', 'unknown')}, skipping sample: {e}")
                return None
        else:
            # No ERA5 data available - return None instead of using synthetic data
            return None
        
        return {
            'past_frames': past_frames,  # (T, C, H, W)
            'future_frames': future_frames,  # (T, C, H, W)
            'past_track': np.stack([past_lons, past_lats], axis=1),  # (T, 2)
            'future_track': np.stack([future_lons, future_lats], axis=1),  # (T, 2)
            'past_intensity': past_winds,  # (T,)
            'future_intensity': future_winds,  # (T,)
            'past_pressure': past_pressures,  # (T,)
            'future_pressure': future_pressures,  # (T,)
            'storm_id': storm_data['storm_id'],
            'storm_name': storm_data['name']
        }
        
    def _generate_synthetic_frames(self,
                                   lons: np.ndarray,
                                   lats: np.ndarray,
                                   winds: np.ndarray,
                                   image_size: Tuple[int, int],
                                   n_channels: int) -> np.ndarray:
        """
        Generate synthetic satellite-like imagery from track data
        
        This is a placeholder. In a real application, you would:
        1. Query satellite data archives (e.g., NOAA, NASA)
        2. Load actual imagery for the storm times
        3. Process and crop around the typhoon center
        """
        T = len(lons)
        H, W = image_size
        
        frames = np.zeros((T, n_channels, H, W), dtype=np.float32)
        
        for t in range(T):
            # Create a synthetic pattern based on wind intensity
            center_y, center_x = H // 2, W // 2
            
            # Create coordinate grids
            y, x = np.ogrid[:H, :W]
            
            # Distance from center
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            
            # Wind-based intensity pattern
            intensity_scale = winds[t] / 70.0  # Normalize by max typhoon wind
            
            for c in range(n_channels):
                # Create different patterns for different channels
                # Simulate different spectral bands
                pattern = intensity_scale * np.exp(-dist / (10 + c * 0.5))
                
                # Add some noise
                noise = np.random.randn(H, W) * 0.1 * intensity_scale
                
                # Add rotation/asymmetry
                theta = np.arctan2(y - center_y, x - center_x)
                spiral = 0.3 * intensity_scale * np.sin(3 * theta + t * 0.5)
                
                frames[t, c] = pattern + noise + spiral
                
        return frames
        
    def create_dataset(self,
                      n_samples: int = 100,
                      start_year: int = 2015,
                      end_year: int = 2023,
                      past_timesteps: int = 12,
                      future_timesteps: int = 8,
                      use_era5: bool = False,
                      download_era5: bool = False,
                      max_storms_download: int = 10,
                      save_path: Optional[str] = None) -> List[Dict]:
        """
        Create a full dataset from IBTrACS WP data
        
        Args:
            n_samples: Number of samples to generate
            start_year: Start year for filtering storms
            end_year: End year for filtering storms
            past_timesteps: Number of past timesteps
            future_timesteps: Number of future timesteps
            use_era5: Whether to use ERA5 data (if available)
            download_era5: Whether to download ERA5 data if not cached
            max_storms_download: Maximum number of storms to download ERA5 data for
            save_path: Path to save the dataset
        
        Returns:
            List of training samples
        """
        print("\n" + "="*80)
        print("CREATING TYPHOON DATASET FROM IBTRACS WP + ERA5")
        print("="*80)
        
        # Load IBTrACS data
        df = self.load_ibtracs()
        
        # Filter for strong typhoons
        storm_ids = self.filter_typhoons(df, start_year, end_year)
        
        if len(storm_ids) == 0:
            raise ValueError("No storms found matching criteria")
        
        # Initialize ERA5 loader if requested
        era5_loader = None
        era5_datasets = {}
        
        if use_era5:
            print("\n" + "-"*80)
            print("ERA5 INTEGRATION")
            print("-"*80)
            era5_loader = ERA5Loader()
            
            if download_era5:
                print(f"Downloading ERA5 data for up to {max_storms_download} storms...")
                storms_to_download = storm_ids[:max_storms_download]
                for idx, storm_id in enumerate(storms_to_download, 1):
                    storm_data = self.get_storm_data(df, storm_id)
                    print(f"\n[{idx}/{len(storms_to_download)}] Processing {storm_data['name']} ({storm_id})...")
                    era5_file = era5_loader.download_era5_for_storm(storm_data)
                    if era5_file:
                        era5_datasets[storm_id] = era5_loader.load_era5(era5_file)
                        print(f"[OK] Loaded ERA5 data for {storm_data['name']}")
            else:
                # Try to load cached ERA5 data
                print("Looking for cached ERA5 data...")
                for storm_id in storm_ids:
                    storm_id_clean = storm_id.replace(' ', '_')
                    era5_file = era5_loader.data_dir / f"{storm_id_clean}_era5.nc"
                    if era5_file.exists():
                        era5_datasets[storm_id] = era5_loader.load_era5(era5_file)
                
                print(f"[OK] Found cached ERA5 data for {len(era5_datasets)} storms")
        
        # Create samples
        samples = []
        print(f"\nGenerating {n_samples} training samples...")
        
        attempts = 0
        max_attempts = n_samples * 10  # Avoid infinite loop
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a storm
            storm_id = np.random.choice(storm_ids)
            
            # Get storm data
            storm_data = self.get_storm_data(df, storm_id)
            
            # Get ERA5 data if available
            era5_dataset = era5_datasets.get(storm_id, None) if use_era5 else None
            
            # Create training sample
            sample = self.create_training_sample(
                storm_data,
                past_timesteps=past_timesteps,
                future_timesteps=future_timesteps,
                era5_dataset=era5_dataset,
                era5_loader=era5_loader,
                use_era5=use_era5
            )
            
            if sample is not None:
                samples.append(sample)
                
                if len(samples) % 10 == 0:
                    print(f"  Generated {len(samples)}/{n_samples} samples...")
        
        print(f"\n[OK] Created {len(samples)} samples from {len(set([s['storm_id'] for s in samples]))} unique storms")
        
        if use_era5:
            n_with_era5 = sum(1 for s in samples if s['storm_id'] in era5_datasets)
            print(f"  {n_with_era5}/{len(samples)} samples use real ERA5 data")
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as compressed numpy
            np.savez_compressed(
                save_path,
                samples=[sample for sample in samples],
                metadata={
                    'n_samples': len(samples),
                    'start_year': start_year,
                    'end_year': end_year,
                    'past_timesteps': past_timesteps,
                    'future_timesteps': future_timesteps,
                    'use_era5': use_era5,
                    'created': datetime.now().isoformat()
                }
            )
            print(f"[OK] Saved dataset to {save_path}")
        
        return samples
        
    def get_storm_summary(self, df: pd.DataFrame, storm_id: str) -> str:
        """Get a summary string for a storm"""
        storm_data = self.get_storm_data(df, storm_id)
        
        max_wind = np.nanmax(storm_data['winds'])
        min_pressure = np.nanmin(storm_data['pressures'])
        duration_hours = len(storm_data['times']) * 6  # Assuming 6-hour intervals
        
        return (f"{storm_data['name']} ({storm_id}): "
                f"Max wind {max_wind:.1f} m/s, "
                f"Min pressure {min_pressure:.0f} hPa, "
                f"Duration {duration_hours:.0f}h")


class ERA5Loader:
    """
    Load ERA5 reanalysis data for typhoon analysis
    
    ERA5 provides global meteorological reanalysis data including:
    - Wind components (u, v)
    - Temperature
    - Geopotential height
    - Relative humidity
    - Mean sea level pressure
    
    Setup:
    1. Register at https://cds.climate.copernicus.eu/
    2. Get API key from your account page
    3. Create ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api/v2
       key: YOUR_UID:YOUR_API_KEY
    """
    
    # ERA5 pressure levels (hPa)
    PRESSURE_LEVELS = ['200', '300', '500', '700', '850', '925', '1000']
    
    # ERA5 variables
    SINGLE_LEVEL_VARS = [
        'mean_sea_level_pressure',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'total_precipitation',
        'total_column_water_vapour',
    ]
    
    PRESSURE_LEVEL_VARS = [
        'geopotential',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'relative_humidity',
        'vertical_velocity',
    ]
    
    # Mapping from actual ERA5 variable names to expected names
    # ERA5 files use short names: z=geopotential, t=temperature, u/v=wind, r=relative_humidity, vo=vertical_velocity
    VAR_NAME_MAPPING = {
        'z': 'geopotential',
        't': 'temperature',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'r': 'relative_humidity',
        'vo': 'vertical_velocity',
        'q': 'specific_humidity',  # Not in standard list but may be present
    }
    
    # Reverse mapping for lookup
    REVERSE_VAR_MAPPING = {v: k for k, v in VAR_NAME_MAPPING.items()}
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        if data_dir is None:
            base_dir = Path(__file__).resolve().parent
            data_dir = base_dir / "data/era5"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_CDSAPI:
            warnings.warn(
                "cdsapi not installed. Install with: pip install cdsapi\n"
                "Also set up your CDS API key in ~/.cdsapirc"
            )
    
    def download_era5_for_storm(self,
                                storm_data: Dict,
                                output_file: Optional[Path] = None,
                                area_size: float = 15.0,  # degrees
                                use_cached: bool = True) -> Optional[Path]:
        """
        Download ERA5 data for a specific storm
        
        Args:
            storm_data: Storm data dictionary from IBTrACS
            output_file: Where to save the downloaded data
            area_size: Size of the area around storm center (degrees)
            use_cached: Use cached data if available
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        if not HAS_CDSAPI:
            return None
        
        # Generate output filename
        if output_file is None:
            storm_id = storm_data['storm_id'].replace(' ', '_')
            output_file = self.data_dir / f"{storm_id}_era5.nc"
        
        if output_file.exists() and use_cached:
            print(f"[OK] Using cached ERA5 data: {output_file}")
            return output_file
        
        # Get time range
        start_time = pd.to_datetime(storm_data['times'][0])
        end_time = pd.to_datetime(storm_data['times'][-1])
        
        # Get spatial extent
        lats = storm_data['lats']
        lons = storm_data['lons']
        
        lat_min = max(-90, float(np.min(lats) - area_size))
        lat_max = min(90, float(np.max(lats) + area_size))
        lon_min = max(-180, float(np.min(lons) - area_size))
        lon_max = min(180, float(np.max(lons) + area_size))
        
        # Format for CDS API
        area = [lat_max, lon_min, lat_min, lon_max]  # North, West, South, East
        
        print(f"\nDownloading ERA5 data for {storm_data['name']}...")
        print(f"  Time: {start_time} to {end_time}")
        print(f"  Area: {area}")
        
        try:
            c = cdsapi.Client()
            
            # Download single-level variables
            print("  Downloading single-level variables...")
            sl_file = output_file.with_suffix('.sl.nc')
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': self.SINGLE_LEVEL_VARS,
                    'year': [str(y) for y in range(start_time.year, end_time.year + 1)],
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(0, 24, 6)],  # 6-hourly
                    'area': area,
                    'format': 'netcdf',
                },
                str(sl_file)
            )
            
            # Download pressure-level variables
            print("  Downloading pressure-level variables...")
            pl_file = output_file.with_suffix('.pl.nc')
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': self.PRESSURE_LEVEL_VARS,
                    'pressure_level': self.PRESSURE_LEVELS,
                    'year': [str(y) for y in range(start_time.year, end_time.year + 1)],
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': [f'{h:02d}:00' for h in range(0, 24, 6)],  # 6-hourly
                    'area': area,
                    'format': 'netcdf',
                },
                str(pl_file)
            )
            
            # Merge the two files
            print("  Merging datasets...")
            ds_sl = xr.open_dataset(sl_file)
            ds_pl = xr.open_dataset(pl_file)
            ds_merged = xr.merge([ds_sl, ds_pl], join='outer')
            ds_merged.to_netcdf(output_file)
            
            # Clean up temporary files
            sl_file.unlink()
            pl_file.unlink()
            
            print(f"[OK] Downloaded ERA5 data to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"[ERROR] Error downloading ERA5 data: {e}")
            print("Make sure you have:")
            print("  1. Registered at https://cds.climate.copernicus.eu/")
            print("  2. Set up ~/.cdsapirc with your API key")
            return None
    
    def load_era5(self, file_path: Path) -> xr.Dataset:
        """Load ERA5 netCDF file"""
        return xr.open_dataset(file_path)
    
    def load_era5_from_daily_files(self,
                                   start_time: pd.Timestamp,
                                   end_time: pd.Timestamp,
                                   lat_range: Tuple[float, float],
                                   lon_range: Tuple[float, float],
                                   max_retries: int = 2,
                                   retry_delay: float = 0.5) -> Optional[xr.Dataset]:
        """
        Load ERA5 data from daily files organized by year
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            lat_range: (lat_min, lat_max) in degrees
            lon_range: (lon_min, lon_max) in degrees
            max_retries: Maximum number of retries for failed file reads
            retry_delay: Delay in seconds between retries
        
        Returns:
            Merged xarray Dataset or None if no data found
        """
        # Suppress dask performance warnings to reduce log noise
        import dask
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='dask')
            
            # Generate list of dates to load
            dates = pd.date_range(start=start_time.date(), end=end_time.date(), freq='D')
            
            datasets = []
        
        for date in dates:
            year = date.year
            date_str = date.strftime('%Y%m%d')
            
            # Look for file in year-specific directory (handle both single and split files)
            year_dir = self.data_dir / f"ERA5_{year}_26data"
            file_path = year_dir / f"era5_pl_{date_str}.nc"
            file_path_part1 = year_dir / f"era5_pl_{date_str}_part1.nc"
            file_path_part2 = year_dir / f"era5_pl_{date_str}_part2.nc"
            
            # Try to load single file first, then try split files
            ds = None
            if file_path.exists():
                # Retry logic for transient HDF5 errors
                for attempt in range(max_retries + 1):
                    try:
                        # Check file size first (corrupted files might be 0 bytes)
                        file_size = file_path.stat().st_size
                        if file_size == 0:
                            if attempt == 0:  # Only print once
                                print(f"  Warning: File is empty (0 bytes): {file_path.name}")
                            break  # Don't retry empty files
                        
                        # Load with chunking to reduce memory usage (if time dimension exists)
                        # Optimized loading - no chunking to avoid dask overhead
                        try:
                            ds = xr.open_dataset(
                                file_path, 
                                engine='h5netcdf',
                                decode_cf=True,
                                mask_and_scale=True,
                                decode_times=True
                            )
                        except (KeyError, ValueError):
                            # No time dimension, load without chunking
                            ds = xr.open_dataset(
                                file_path,
                                engine='h5netcdf',
                                decode_cf=True,
                                mask_and_scale=True,
                                decode_times=True
                            )
                        
                        # Success - break out of retry loop
                        break
                        
                    except (OSError, IOError) as e:
                        # HDF5/netCDF file corruption or I/O error
                        if attempt < max_retries:
                            if attempt == 0:  # Only print on first attempt
                                print(f"  Warning: File I/O error for {file_path.name}, retrying...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(f"  Warning: File I/O error for {file_path.name} after {max_retries + 1} attempts: {str(e)[:100]}")
                            print(f"    Skipping this file...")
                            ds = None
                            break
                    except Exception as e:
                        # Other errors - don't retry
                        if attempt == 0:  # Only print once
                            print(f"  Warning: Failed to load {file_path.name}: {str(e)[:100]}")
                        ds = None
                        break
                
                if ds is None:
                    continue
            elif file_path_part1.exists() and file_path_part2.exists():
                # Retry logic for split files
                ds_part1 = None
                ds_part2 = None
                for attempt in range(max_retries + 1):
                    try:
                        # Check file sizes first
                        size1 = file_path_part1.stat().st_size
                        size2 = file_path_part2.stat().st_size
                        if size1 == 0 or size2 == 0:
                            if attempt == 0:
                                print(f"  Warning: One or both split files are empty: {file_path_part1.name}, {file_path_part2.name}")
                            break
                        
                        # Load both parts and merge them - no chunking for speed
                        try:
                            ds_part1 = xr.open_dataset(
                                file_path_part1, 
                                engine='h5netcdf',
                                decode_cf=True,
                                mask_and_scale=True
                            )
                            ds_part2 = xr.open_dataset(
                                file_path_part2, 
                                engine='h5netcdf',
                                decode_cf=True,
                                mask_and_scale=True
                            )
                        except (KeyError, ValueError):
                            # No time dimension, load without chunking
                            ds_part1 = xr.open_dataset(file_path_part1, engine='h5netcdf', decode_cf=True, mask_and_scale=True)
                            ds_part2 = xr.open_dataset(file_path_part2, engine='h5netcdf', decode_cf=True, mask_and_scale=True)
                        
                        # Merge along the appropriate dimension (usually longitude or time)
                        # Try merging along longitude first (common for split files)
                        if 'longitude' in ds_part1.dims and 'longitude' in ds_part2.dims:
                            ds = xr.concat([ds_part1, ds_part2], dim='longitude', join='outer')
                        elif 'time' in ds_part1.dims and 'time' in ds_part2.dims:
                            ds = xr.concat([ds_part1, ds_part2], dim='time', join='outer')
                        else:
                            # If dimensions don't match, try merge
                            ds = xr.merge([ds_part1, ds_part2], join='outer')
                        
                        # Success - break out of retry loop
                        break
                        
                    except (OSError, IOError) as e:
                        # Clean up on error
                        if ds_part1 is not None:
                            try:
                                ds_part1.close()
                            except:
                                pass
                        if ds_part2 is not None:
                            try:
                                ds_part2.close()
                            except:
                                pass
                        ds_part1 = None
                        ds_part2 = None
                        
                        if attempt < max_retries:
                            if attempt == 0:
                                print(f"  Warning: File I/O error for split files, retrying...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            print(f"  Warning: File I/O error for split files after {max_retries + 1} attempts: {str(e)[:100]}")
                            print(f"    Skipping these files...")
                            ds = None
                            break
                    except Exception as e:
                        # Clean up on error
                        if ds_part1 is not None:
                            try:
                                ds_part1.close()
                            except:
                                pass
                        if ds_part2 is not None:
                            try:
                                ds_part2.close()
                            except:
                                pass
                        if attempt == 0:
                            print(f"  Warning: Failed to load/merge split files: {str(e)[:100]}")
                        ds = None
                        break
                    finally:
                        # Always close intermediate datasets to free memory
                        if ds_part1 is not None and ds is not None:
                            try:
                                ds_part1.close()
                            except:
                                pass
                        if ds_part2 is not None and ds is not None:
                            try:
                                ds_part2.close()
                            except:
                                pass
                
                if ds is None:
                    continue
            
            if ds is not None:
                try:
                    # Check if file covers the requested spatial region
                    # NOTE: Files may be pre-cropped, so we check coverage first
                    if 'latitude' in ds.coords and 'longitude' in ds.coords:
                        lat_min, lat_max = lat_range
                        lon_min, lon_max = lon_range
                        
                        # Get file's actual coordinate ranges
                        file_lat_coords = ds.coords['latitude'].values
                        file_lon_coords = ds.coords['longitude'].values
                        file_lat_min = float(file_lat_coords.min())
                        file_lat_max = float(file_lat_coords.max())
                        file_lon_min = float(file_lon_coords.min())
                        file_lon_max = float(file_lon_coords.max())
                        
                        # Check if requested range overlaps with file range
                        # The file must cover at least part of the requested region
                        lat_overlap = not (lat_max < file_lat_min or lat_min > file_lat_max)
                        lon_overlap = not (lon_max < file_lon_min or lon_min > file_lon_max)
                        
                        if not lat_overlap or not lon_overlap:
                            # No overlap - this file doesn't cover the requested region
                            # This can happen if files are pre-cropped to a small region
                            ds.close()
                            continue
                        
                        # If file is already cropped (smaller than requested), use it as-is
                        # Otherwise, crop to requested region
                        file_covers_requested = (
                            file_lat_min <= lat_min and file_lat_max >= lat_max and
                            file_lon_min <= lon_min and file_lon_max >= lon_max
                        )
                        
                        if not file_covers_requested:
                            # File is smaller than requested - use it as-is (it's pre-cropped)
                            # The extraction function will handle cropping to storm location
                            pass  # Don't crop further, use file as-is
                        else:
                            # File is larger than requested - crop to requested region
                            # Clamp requested range to file's available range
                            lat_min_clamped = max(lat_min, file_lat_min)
                            lat_max_clamped = min(lat_max, file_lat_max)
                            lon_min_clamped = max(lon_min, file_lon_min)
                            lon_max_clamped = min(lon_max, file_lon_max)
                            
                            # Handle longitude wrapping if needed
                            if lon_min_clamped < -180:
                                lon_min_clamped = -180
                            if lon_max_clamped > 180:
                                lon_max_clamped = 180
                            
                            # Check latitude order (ERA5 typically descending)
                            if len(file_lat_coords) > 1 and file_lat_coords[0] > file_lat_coords[-1]:
                                # Descending order (north to south) - slice from high to low
                                ds = ds.sel(
                                    latitude=slice(lat_max_clamped, lat_min_clamped),
                                    longitude=slice(lon_min_clamped, lon_max_clamped)
                                )
                            else:
                                # Ascending order (south to north) - slice from low to high
                                ds = ds.sel(
                                    latitude=slice(lat_min_clamped, lat_max_clamped),
                                    longitude=slice(lon_min_clamped, lon_max_clamped)
                                )
                            
                            # Check if selection resulted in empty dataset
                            if 'longitude' in ds.dims and ds.dims['longitude'] == 0:
                                ds.close()
                                continue
                            if 'latitude' in ds.dims and ds.dims['latitude'] == 0:
                                ds.close()
                                continue
                    
                    datasets.append(ds)
                except Exception as e:
                    if ds is not None:
                        ds.close()
                    # Suppress warnings for files that don't cover the region (expected for pre-cropped files)
                    continue
        
        if not datasets:
            return None
        
        # Merge all daily datasets
        try:
            if not datasets:
                return None
                
            # Determine time dimension name
            time_dim = None
            for dim_name in ['time', 'valid_time', 't']:
                if dim_name in datasets[0].dims or dim_name in datasets[0].coords:
                    time_dim = dim_name
                    break
            
            if time_dim is not None:
                # Use 'outer' join to create union of all coordinates
                # This is necessary because files are cropped per date and may have different spatial coverage
                # The extraction function will check coverage per timestep
                # Note: 'outer' join will fill missing values with NaN automatically
                merged_ds = xr.concat(datasets, dim=time_dim, join='outer')
            else:
                # No time dimension - try merge instead
                merged_ds = xr.merge(datasets, join='outer')
            
            # Close individual datasets after merging to free memory
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass
            
            return merged_ds
        except Exception as e:
            # Clean up all datasets on error
            for ds in datasets:
                try:
                    ds.close()
                except:
                    pass
            # If concat fails, try merge
            try:
                merged_ds = xr.merge(datasets, join='outer')
                return merged_ds
            except Exception as e2:
                print(f"  Warning: Failed to merge ERA5 datasets: {e2}")
                # Return first dataset if merge fails
                return datasets[0] if datasets else None
    
    def extract_frames_at_times(self,
                               ds: xr.Dataset,
                               center_lons: np.ndarray,
                               center_lats: np.ndarray,
                               times: np.ndarray,
                               crop_size: int = 64,  # Default 64x64 for better spatial resolution
                               resolution: float = 0.25,
                               load_per_timestep: bool = True) -> np.ndarray:
        """
        Extract cropped frames centered on storm at each timestep
        
        Args:
            ds: ERA5 xarray Dataset (can be None if load_per_timestep=True)
            center_lons: Storm center longitudes
            center_lats: Storm center latitudes
            times: Timestamps
            crop_size: Size of cropped region in pixels (None = use actual file size)
            resolution: ERA5 resolution in degrees (0.25° for ERA5)
            load_per_timestep: If True, load ERA5 file separately for each timestep (avoids NaN from merging)
            
        Returns:
            Array of shape (T, C, H, W) where C is number of variables
        """
        T = len(times)
        
        # If load_per_timestep, we'll load individual files for each timestep
        if load_per_timestep:
            return self._extract_frames_per_timestep(
                center_lons=center_lons,
                center_lats=center_lats,
                times=times,
                crop_size=crop_size,
                resolution=resolution
            )
        
        # Original merged dataset logic (kept for backwards compatibility)
        # Check if crop_size is valid for the file
        if ds is not None and 'latitude' in ds.coords and 'longitude' in ds.coords:
            lat_size = ds.sizes.get('latitude', 0)
            lon_size = ds.sizes.get('longitude', 0)
            actual_size = min(lat_size, lon_size)
            
            # Ensure crop_size doesn't exceed file size
            if crop_size > actual_size:
                crop_size = actual_size
                print(f"Warning: Requested crop_size {crop_size} exceeds file size {actual_size}, using {crop_size}")
        # If crop_size not specified, use 64 as default
        if crop_size is None:
            crop_size = 64
        
        # Calculate spatial extent for cropping
        box_size_deg = crop_size * resolution / 2
        
        # Determine time dimension name
        time_dim = None
        for dim_name in ['time', 'valid_time', 't']:
            if dim_name in ds.dims or dim_name in ds.coords:
                time_dim = dim_name
                break
        
        all_frames = []
        
        for t in range(T):
            center_lon = center_lons[t]
            center_lat = center_lats[t]
            time = pd.to_datetime(times[t])
            
            # Define crop region
            # Latitude: ERA5 uses descending order (north to south), so slice from high to low
            # Longitude: ascending order (west to east)
            lat_min = center_lat - box_size_deg
            lat_max = center_lat + box_size_deg
            lon_min = center_lon - box_size_deg
            lon_max = center_lon + box_size_deg
            
            # Extract data
            try:
                # Select nearest time if time dimension exists
                if time_dim is not None:
                    try:
                        ds_t = ds.sel({time_dim: time}, method='nearest')
                    except (KeyError, ValueError) as e:
                        # If time selection fails, try using isel with nearest index
                        time_coords = ds.coords[time_dim]
                        if hasattr(time_coords, 'values'):
                            time_values = pd.to_datetime(time_coords.values)
                            time_idx = np.argmin(np.abs((time_values - time).total_seconds()))
                            ds_t = ds.isel({time_dim: time_idx})
                        else:
                            raise e
                else:
                    # No time dimension - use dataset as-is (might be single time slice)
                    ds_t = ds
                
                # Check if storm location is within the file's spatial coverage
                # CRITICAL: Files are cropped per date to match typhoon locations
                # When merged, different timesteps may have different spatial coverage
                # So we must check coverage for THIS specific timestep
                if 'longitude' in ds_t.coords and 'latitude' in ds_t.coords:
                    lon_coords = ds_t.coords['longitude'].values
                    lat_coords = ds_t.coords['latitude'].values
                    
                    # Handle case where coordinates might be 1D arrays or multi-dimensional
                    if lon_coords.ndim > 1:
                        # Multi-dimensional - get actual coordinate values
                        file_lon_min = float(np.nanmin(lon_coords))
                        file_lon_max = float(np.nanmax(lon_coords))
                        file_lat_min = float(np.nanmin(lat_coords))
                        file_lat_max = float(np.nanmax(lat_coords))
                    else:
                        # 1D coordinate arrays
                        file_lon_min = float(lon_coords.min())
                        file_lon_max = float(lon_coords.max())
                        file_lat_min = float(lat_coords.min())
                        file_lat_max = float(lat_coords.max())
                    
                    # Check if storm center is within file coverage for this timestep
                    center_in_file = (
                        file_lon_min <= center_lon <= file_lon_max and
                        file_lat_min <= center_lat <= file_lat_max
                    )
                    
                    if not center_in_file:
                        # Storm center is outside file coverage for this timestep
                        # This can happen when files are cropped per date and typhoon moves
                        # Use nearest available data point
                        nearest_lat = np.clip(center_lat, file_lat_min, file_lat_max)
                        nearest_lon = np.clip(center_lon, file_lon_min, file_lon_max)
                        center_lat = nearest_lat
                        center_lon = nearest_lon
                        lat_min = center_lat - box_size_deg
                        lat_max = center_lat + box_size_deg
                        lon_min = center_lon - box_size_deg
                        lon_max = center_lon + box_size_deg
                    
                    # Also check if crop region overlaps
                    crop_overlaps = not (
                        lon_max < file_lon_min or lon_min > file_lon_max or
                        lat_max < file_lat_min or lat_min > file_lat_max
                    )
                    
                    if not crop_overlaps:
                        # Crop region doesn't overlap - clip to file bounds
                        lat_min = max(lat_min, file_lat_min)
                        lat_max = min(lat_max, file_lat_max)
                        lon_min = max(lon_min, file_lon_min)
                        lon_max = min(lon_max, file_lon_max)
                
                # Crop region - check latitude coordinate order
                if 'latitude' in ds_t.coords:
                    lat_coords = ds_t.coords['latitude'].values
                    if len(lat_coords) > 1 and lat_coords[0] > lat_coords[-1]:
                        # Descending order (north to south) - slice from high to low
                        lat_slice = slice(lat_max, lat_min)
                    else:
                        # Ascending order (south to north) - slice from low to high
                        lat_slice = slice(lat_min, lat_max)
                else:
                    lat_slice = slice(lat_max, lat_min)  # Default to descending
                
                lon_slice = slice(lon_min, lon_max)
                ds_crop = ds_t.sel(latitude=lat_slice, longitude=lon_slice)
                
                # Check if crop is empty
                if 'longitude' in ds_crop.dims and ds_crop.dims['longitude'] == 0:
                    raise ValueError(f"Empty longitude selection: requested [{lon_min:.2f}, {lon_max:.2f}], file has [{file_lon_min:.2f}, {file_lon_max:.2f}]")
                if 'latitude' in ds_crop.dims and ds_crop.dims['latitude'] == 0:
                    raise ValueError(f"Empty latitude selection: requested [{lat_min:.2f}, {lat_max:.2f}], file has [{file_lat_min:.2f}, {file_lat_max:.2f}]")
                
                # Check if we got valid data (not all NaN) - this happens when merged dataset has coordinates
                # but no data for this timestep/location
                if len(ds_crop.data_vars) > 0:
                    # Check first variable to see if we have actual data
                    first_var = list(ds_crop.data_vars.keys())[0]
                    data_sample = ds_crop[first_var].values
                    if np.all(np.isnan(data_sample)):
                        raise ValueError(f"All data is NaN for timestep {t} at location ({center_lat:.2f}°N, {center_lon:.2f}°E) - file doesn't have data for this location")
                
                # Stack all variables into channels
                channels = []
                
                # Get actual variable names in the dataset
                actual_vars = list(ds_crop.data_vars.keys())
                
                # DEBUG: Print what we're working with
                # print(f"DEBUG: Actual vars in cropped dataset: {actual_vars}")
                # print(f"DEBUG: Expected single-level: {self.SINGLE_LEVEL_VARS}")
                # print(f"DEBUG: Expected pressure-level: {self.PRESSURE_LEVEL_VARS}")
                
                # Single-level variables (check both expected names and mapped names)
                # NOTE: Your ERA5 files may only have pressure-level data, so single-level vars might not exist
                for var in self.SINGLE_LEVEL_VARS:
                    # Try expected name first
                    if var in ds_crop:
                        data = ds_crop[var].values
                        # Handle multi-dimensional data (remove time/level dims if present)
                        if data.ndim > 2:
                            # Take first slice if time dimension still exists
                            data = data.reshape(-1, *data.shape[-2:])[0]
                        # Resize to target size if needed
                        if data.shape != (crop_size, crop_size):
                            data = self._resize_to_target(data, crop_size)
                        channels.append(data)
                    else:
                        # Try reverse mapping (short name) - but single-level vars don't have short names in your files
                        short_name = self.REVERSE_VAR_MAPPING.get(var)
                        if short_name and short_name in ds_crop:
                            data = ds_crop[short_name].values
                            if data.ndim > 2:
                                data = data.reshape(-1, *data.shape[-2:])[0]
                            if data.shape != (crop_size, crop_size):
                                data = self._resize_to_target(data, crop_size)
                            channels.append(data)
                        # If not found, skip (single-level vars may not be in pressure-level files)
                
                # Pressure-level variables (one channel per level)
                for var in self.PRESSURE_LEVEL_VARS:
                    # Try expected name first
                    if var in ds_crop:
                        for level in self.PRESSURE_LEVELS:
                            try:
                                data = ds_crop[var].sel(level=int(level)).values
                                # Handle multi-dimensional data
                                if data.ndim > 2:
                                    data = data.reshape(-1, *data.shape[-2:])[0]
                                if data.shape != (crop_size, crop_size):
                                    data = self._resize_to_target(data, crop_size)
                                channels.append(data)
                            except:
                                pass
                    else:
                        # Try reverse mapping (short name)
                        short_name = self.REVERSE_VAR_MAPPING.get(var)
                        if short_name and short_name in ds_crop:
                            # Get actual pressure levels from the dataset
                            if 'pressure_level' in ds_crop[short_name].dims:
                                actual_levels = ds_crop[short_name].coords['pressure_level'].values
                            elif 'level' in ds_crop[short_name].dims:
                                actual_levels = ds_crop[short_name].coords['level'].values
                            else:
                                actual_levels = []
                            
                            # Use actual levels if available, otherwise try expected levels
                            levels_to_use = actual_levels if len(actual_levels) > 0 else [int(l) for l in self.PRESSURE_LEVELS]
                            
                            for level in levels_to_use:
                                try:
                                    # Check if level dimension exists (might be pressure_level)
                                    if 'pressure_level' in ds_crop[short_name].dims:
                                        data = ds_crop[short_name].sel(pressure_level=float(level), method='nearest').values
                                    elif 'level' in ds_crop[short_name].dims:
                                        data = ds_crop[short_name].sel(level=float(level), method='nearest').values
                                    else:
                                        # No level dimension, use as-is
                                        data = ds_crop[short_name].values
                                    
                                    # Handle multi-dimensional data
                                    if data.ndim > 2:
                                        data = data.reshape(-1, *data.shape[-2:])[0]
                                    if data.shape != (crop_size, crop_size):
                                        data = self._resize_to_target(data, crop_size)
                                    channels.append(data)
                                except Exception as e:
                                    pass
                
                # Stack into (C, H, W)
                if channels:
                    frame = np.stack(channels, axis=0)
                    # Ensure consistent shape - if crop_size was auto-detected, all frames should have same size
                    all_frames.append(frame)
                else:
                    # No channels extracted - this means the data extraction failed
                    raise ValueError(f"No ERA5 variables extracted for timestep {t} at location ({center_lat:.2f}°N, {center_lon:.2f}°E) - dataset may be empty or missing variables")
                
            except Exception as e:
                # Extraction failed for this timestep - raise exception to fail the entire extraction
                # This ensures we don't silently return zeros
                raise ValueError(f"Failed to extract ERA5 frame at timestep {t} (time={times[t]}, location=({center_lat:.2f}°N, {center_lon:.2f}°E)): {e}")
        
        if not all_frames:
            # If no frames extracted, raise exception instead of returning zeros
            raise ValueError(f"Failed to extract any ERA5 frames for {T} timesteps")
        
        # Ensure all frames have the same spatial dimensions
        # This handles cases where different timesteps might have slightly different sizes
        if all_frames:
            # Get target shape from first frame
            target_shape = all_frames[0].shape[1:]  # (H, W)
            for i, frame in enumerate(all_frames):
                if frame.shape[1:] != target_shape:
                    # Resize to match first frame
                    from scipy.ndimage import zoom
                    h_ratio = target_shape[0] / frame.shape[1]
                    w_ratio = target_shape[1] / frame.shape[2]
                    # Resize each channel
                    resized_channels = []
                    for c in range(frame.shape[0]):
                        resized = zoom(frame[c], (h_ratio, w_ratio), order=1)
                        resized_channels.append(resized)
                    all_frames[i] = np.stack(resized_channels, axis=0)
        
        return np.stack(all_frames, axis=0)  # (T, C, H, W)
    
    def _extract_frames_per_timestep(self,
                                    center_lons: np.ndarray,
                                    center_lats: np.ndarray,
                                    times: np.ndarray,
                                    crop_size: int = 64,
                                    resolution: float = 0.25) -> np.ndarray:
        """
        Extract frames by loading individual ERA5 files for each timestep.
        Avoids NaN issues from merging files with different spatial coverage.
        """
        T = len(times)
        all_frames = []
        box_size_deg = crop_size * resolution / 2
        
        for t in range(T):
            center_lon = center_lons[t]
            center_lat = center_lats[t]
            time = pd.to_datetime(times[t])
            
            # Load ERA5 file for this specific date
            date_str = time.strftime('%Y%m%d')
            year = time.year
            year_dir = self.data_dir / f"ERA5_{year}_26data"
            file_path = year_dir / f"era5_pl_{date_str}.nc"
            
            if not file_path.exists():
                raise ValueError(f"ERA5 file not found for timestep {t} (date={date_str}): {file_path}")
            
            try:
                # Load single file for this date
                ds = xr.open_dataset(file_path, engine='h5netcdf', decode_cf=True, mask_and_scale=True, decode_times=True)
                
                # Select time closest to target
                time_dim = None
                for dim_name in ['time', 'valid_time', 't']:
                    if dim_name in ds.dims or dim_name in ds.coords:
                        time_dim = dim_name
                        break
                
                if time_dim is not None:
                    try:
                        ds_t = ds.sel({time_dim: time}, method='nearest')
                    except:
                        time_coords = ds.coords[time_dim]
                        time_values = pd.to_datetime(time_coords.values)
                        time_idx = np.argmin(np.abs((time_values - time).total_seconds()))
                        ds_t = ds.isel({time_dim: time_idx})
                else:
                    ds_t = ds
                
                # Check spatial coverage
                if 'longitude' in ds_t.coords and 'latitude' in ds_t.coords:
                    lon_coords = ds_t.coords['longitude'].values
                    lat_coords = ds_t.coords['latitude'].values
                    
                    file_lon_min = float(lon_coords.min())
                    file_lon_max = float(lon_coords.max())
                    file_lat_min = float(lat_coords.min())
                    file_lat_max = float(lat_coords.max())
                    
                    # Check if storm center is within file coverage
                    if not (file_lon_min <= center_lon <= file_lon_max and file_lat_min <= center_lat <= file_lat_max):
                        ds.close()
                        raise ValueError(f"Storm location ({center_lat:.2f}°N, {center_lon:.2f}°E) outside ERA5 file coverage ([{file_lat_min:.2f}, {file_lat_max:.2f}]°N, [{file_lon_min:.2f}, {file_lon_max:.2f}]°E) for date {date_str}")
                
                # Define crop region
                lat_min = center_lat - box_size_deg
                lat_max = center_lat + box_size_deg
                lon_min = center_lon - box_size_deg
                lon_max = center_lon + box_size_deg
                
                # Clip to file bounds
                lat_min = max(lat_min, file_lat_min)
                lat_max = min(lat_max, file_lat_max)
                lon_min = max(lon_min, file_lon_min)
                lon_max = min(lon_max, file_lon_max)
                
                # Handle latitude coordinate order
                if len(lat_coords) > 1 and lat_coords[0] > lat_coords[-1]:
                    lat_slice = slice(lat_max, lat_min)  # Descending
                else:
                    lat_slice = slice(lat_min, lat_max)  # Ascending
                
                lon_slice = slice(lon_min, lon_max)
                ds_crop = ds_t.sel(latitude=lat_slice, longitude=lon_slice)
                
                # Extract variables into channels
                channels = []
                
                # Pressure-level variables (we know your files have these)
                for var in self.PRESSURE_LEVEL_VARS:
                    short_name = self.REVERSE_VAR_MAPPING.get(var)
                    if short_name and short_name in ds_crop:
                        if 'pressure_level' in ds_crop[short_name].dims:
                            actual_levels = ds_crop[short_name].coords['pressure_level'].values
                            for level in actual_levels:
                                try:
                                    data = ds_crop[short_name].sel(pressure_level=float(level), method='nearest').values
                                    if data.ndim > 2:
                                        data = data.reshape(-1, *data.shape[-2:])[0]
                                    if data.shape != (crop_size, crop_size):
                                        data = self._resize_to_target(data, crop_size)
                                    channels.append(data)
                                except:
                                    pass
                
                ds.close()
                
                if channels:
                    frame = np.stack(channels, axis=0)  # (C, H, W)
                    all_frames.append(frame)
                else:
                    raise ValueError(f"No variables extracted for timestep {t}")
                    
            except Exception as e:
                raise ValueError(f"Failed to extract frame at timestep {t} (date={date_str}, location=({center_lat:.2f}°N, {center_lon:.2f}°E)): {e}")
        
        if not all_frames:
            raise ValueError(f"Failed to extract any frames for {T} timesteps")
        
        # Ensure consistent shapes
        target_shape = all_frames[0].shape
        for i, frame in enumerate(all_frames):
            if frame.shape != target_shape:
                # Resize if needed
                from scipy.ndimage import zoom
                h_ratio = target_shape[1] / frame.shape[1]
                w_ratio = target_shape[2] / frame.shape[2]
                resized_channels = []
                for c in range(frame.shape[0]):
                    resized = zoom(frame[c], (h_ratio, w_ratio), order=1)
                    resized_channels.append(resized)
                all_frames[i] = np.stack(resized_channels, axis=0)
        
        return np.stack(all_frames, axis=0)  # (T, C, H, W)
    
    def _resize_to_target(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """Resize data to target size using interpolation"""
        from scipy.ndimage import zoom
        
        if data.shape == (target_size, target_size):
            return data
        
        zoom_factors = (target_size / data.shape[0], target_size / data.shape[1])
        return zoom(data, zoom_factors, order=1)
    
    def get_n_channels(self) -> int:
        """Get total number of ERA5 channels"""
        return len(self.SINGLE_LEVEL_VARS) + len(self.PRESSURE_LEVEL_VARS) * len(self.PRESSURE_LEVELS)


def download_and_prepare_real_data(use_era5: bool = False, download_era5: bool = False):
    """
    Download and prepare real typhoon data from IBTrACS WP and optionally ERA5
    
    Args:
        use_era5: Whether to use ERA5 meteorological data
        download_era5: Whether to download ERA5 data (requires CDS API key)
    
    Returns:
        List of training samples
    """
    loader = IBTrACSLoader(data_dir="data/raw")
    
    # Create dataset
    samples = loader.create_dataset(
        n_samples=100,
        start_year=2018,
        end_year=2023,
        past_timesteps=12,
        future_timesteps=8,
        use_era5=use_era5,
        download_era5=download_era5,
        save_path="data/processed/real_typhoons_wp_era5.npz" if use_era5 else "data/processed/real_typhoons_wp.npz"
    )
    
    return samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare IBTrACS WP + ERA5 data')
    parser.add_argument('--use-era5', action='store_true', help='Use ERA5 meteorological data')
    parser.add_argument('--download-era5', action='store_true', help='Download ERA5 data (requires CDS API)')
    parser.add_argument('--demo', action='store_true', help='Run quick demo without ERA5')
    args = parser.parse_args()
    
    if args.demo:
        print("\n" + "="*80)
        print("DEMO MODE: Using IBTrACS WP only (synthetic frames)")
        print("="*80)
        use_era5 = False
        download_era5 = False
    else:
        use_era5 = args.use_era5
        download_era5 = args.download_era5
    
    if use_era5 and not HAS_CDSAPI:
        print("\n" + "!"*80)
        print("WARNING: cdsapi not installed!")
        print("!"*80)
        print("\nTo use ERA5 data:")
        print("1. Install cdsapi: pip install cdsapi")
        print("2. Register at: https://cds.climate.copernicus.eu/")
        print("3. Create ~/.cdsapirc with your API key")
        print("\nFalling back to synthetic frames...")
        use_era5 = False
    
    # Download and prepare data
    samples = download_and_prepare_real_data(use_era5=use_era5, download_era5=download_era5)
    
    print(f"\n{'='*80}")
    print("SAMPLE DATA EXAMPLE")
    print("="*80)
    if samples:
        sample = samples[0]
        print(f"Storm: {sample['storm_name']} ({sample['storm_id']})")
        print(f"Past frames shape: {sample['past_frames'].shape}")
        print(f"Future frames shape: {sample['future_frames'].shape}")
        print(f"Past track shape: {sample['past_track'].shape}")
        print(f"Future track shape: {sample['future_track'].shape}")
        print(f"Past intensity: {sample['past_intensity']}")
        print(f"Future intensity: {sample['future_intensity']}")
        
        n_channels = sample['past_frames'].shape[1]
        if use_era5:
            print(f"\nUsing {n_channels} ERA5 channels:")
            print(f"  - 6 single-level variables")
            print(f"  - 6 pressure-level variables × 7 levels = 42 channels")
        else:
            print(f"\nUsing {n_channels} synthetic channels")

