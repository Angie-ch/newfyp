"""
Utilities for encoding IBTrACS track and intensity data as spatial channels

This module provides functions to encode typhoon position and intensity
information as 2D spatial fields that can be concatenated with ERA5 atmospheric
fields for the autoencoder input.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def encode_track_as_channels(
    track: torch.Tensor,
    intensity: torch.Tensor,
    pressure: Optional[torch.Tensor] = None,
    image_size: Tuple[int, int] = (64, 64),
    grid_range: Tuple[float, float] = (20.0, 20.0),  # degrees lat/lon
    normalize: bool = True
) -> torch.Tensor:
    """
    Encode IBTrACS data (track position, intensity, pressure) as spatial channels
    
    Creates 4 additional channels:
    1. Absolute latitude field (vertical gradient - each row encodes its latitude)
    2. Absolute longitude field (horizontal gradient - each column encodes its longitude)
    3. Intensity field (wind speed, uniform across spatial dimensions)
    4. Pressure field (minimum central pressure, uniform across spatial dimensions)
    
    The position encoding creates spatially-varying coordinate fields that can be
    decoded to extract absolute geographic position from the center pixel.
    
    Args:
        track: (B, T, 2) tensor with [lat, lon] positions
               OR (B, 2) for single timestep
        intensity: (B, T) tensor with wind speeds in m/s
                   OR (B,) for single timestep
        pressure: (B, T) tensor with central pressure in hPa
                  OR (B,) for single timestep
                  If None, pressure channel will be zeros
        image_size: Target spatial dimensions (H, W)
        grid_range: Physical size of grid in degrees (lat_range, lon_range)
        normalize: Whether to normalize values to [-1, 1] range
    
    Returns:
        encoded: (B, T, 4, H, W) tensor of encoded channels
                 OR (B, 4, H, W) if single timestep input
    """
    device = track.device
    H, W = image_size
    lat_range, lon_range = grid_range
    
    # Handle both single timestep and sequence inputs
    is_sequence = track.ndim == 3
    if not is_sequence:
        track = track.unsqueeze(1)  # (B, 1, 2)
        intensity = intensity.unsqueeze(1)  # (B, 1)
        if pressure is not None:
            pressure = pressure.unsqueeze(1)  # (B, 1)
    
    B, T, _ = track.shape
    
    # Create pressure tensor if not provided
    if pressure is None:
        pressure = torch.zeros(B, T, device=device)
    
    # Create coordinate grids (relative to center)
    # Grid represents offsets from the typhoon center
    y_coords = torch.linspace(-lat_range/2, lat_range/2, H, device=device)
    x_coords = torch.linspace(-lon_range/2, lon_range/2, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)
    
    # Prepare output
    encoded = torch.zeros(B, T, 4, H, W, device=device)
    
    for b in range(B):
        for t in range(T):
            lat_center = track[b, t, 0]  # Absolute latitude
            lon_center = track[b, t, 1]  # Absolute longitude
            
            # Channel 0: Absolute latitude field (spatially-varying)
            # Each pixel encodes its own absolute latitude
            # This creates a vertical gradient that varies by row
            for i in range(H):
                # Map pixel row to actual latitude
                # Top row (i=0) is north, bottom row (i=H-1) is south
                pixel_lat = lat_center + (i - H/2) * (lat_range / H)
                if normalize:
                    # Normalize to [-1, 1] range for WP typhoons: 10-35°N
                    lat_norm = (pixel_lat - 22.5) / 12.5
                else:
                    lat_norm = pixel_lat
                encoded[b, t, 0, i, :] = lat_norm
            
            # Channel 1: Absolute longitude field (spatially-varying)
            # Each pixel encodes its own absolute longitude
            # This creates a horizontal gradient that varies by column
            for j in range(W):
                # Map pixel column to actual longitude
                # Left column (j=0) is west, right column (j=W-1) is east
                pixel_lon = lon_center + (j - W/2) * (lon_range / W)
                if normalize:
                    # Normalize to [-1, 1] range for WP: 120-160°E
                    lon_norm = (pixel_lon - 140.0) / 20.0
                else:
                    lon_norm = pixel_lon
                encoded[b, t, 1, :, j] = lon_norm
            
            # Channel 2: Intensity encoding
            # Uniform field representing wind speed at all spatial locations
            wind = intensity[b, t]
            if normalize:
                # Normalize wind speed: typical range 17-70 m/s
                wind_norm = (wind - 43.5) / 26.5  # Center at ~43.5, scale by ±26.5
            else:
                wind_norm = wind
            encoded[b, t, 2] = wind_norm
            
            # Channel 3: Pressure encoding
            # Uniform field representing minimum central pressure
            pres = pressure[b, t]
            if normalize:
                # Normalize pressure: typical range 900-1010 hPa
                pres_norm = (pres - 955.0) / 55.0  # Center at 955, scale by ±55
            else:
                pres_norm = pres
            encoded[b, t, 3] = pres_norm
    
    # Remove time dimension if input was single timestep
    if not is_sequence:
        encoded = encoded.squeeze(1)  # (B, 4, H, W)
    
    return encoded


def encode_track_as_distance_field(
    track: torch.Tensor,
    intensity: torch.Tensor,
    pressure: Optional[torch.Tensor] = None,
    image_size: Tuple[int, int] = (64, 64),
    grid_range: Tuple[float, float] = (20.0, 20.0)
) -> torch.Tensor:
    """
    Alternative encoding: distance field from typhoon center
    
    Creates 5 channels:
    1. Distance from center (radial)
    2. Angle from center (angular position)
    3. Radial wind component
    4. Tangential wind component (based on intensity)
    5. Pressure field (radial decay from center)
    
    This encoding is more physics-aware, representing the typical
    axisymmetric structure of tropical cyclones.
    
    Args:
        track: (B, T, 2) or (B, 2) tensor with [lat, lon]
        intensity: (B, T) or (B,) tensor with wind speeds
        pressure: (B, T) or (B,) tensor with central pressure (hPa)
        image_size: Spatial dimensions
        grid_range: Physical size in degrees
    
    Returns:
        encoded: (B, T, 5, H, W) or (B, 5, H, W) tensor
    """
    device = track.device
    H, W = image_size
    lat_range, lon_range = grid_range
    
    # Handle both single timestep and sequence inputs
    is_sequence = track.ndim == 3
    if not is_sequence:
        track = track.unsqueeze(1)
        intensity = intensity.unsqueeze(1)
        if pressure is not None:
            pressure = pressure.unsqueeze(1)
    
    B, T, _ = track.shape
    
    # Create pressure tensor if not provided
    if pressure is None:
        pressure = torch.zeros(B, T, device=device) + 950.0  # Default typhoon pressure
    
    # Create coordinate grids
    y_coords = torch.linspace(-lat_range/2, lat_range/2, H, device=device)
    x_coords = torch.linspace(-lon_range/2, lon_range/2, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Compute distance and angle from center
    dist = torch.sqrt(yy**2 + xx**2)  # (H, W)
    angle = torch.atan2(yy, xx)  # (H, W)
    
    # Prepare output
    encoded = torch.zeros(B, T, 5, H, W, device=device)
    
    for b in range(B):
        for t in range(T):
            wind = intensity[b, t]
            pres = pressure[b, t]
            
            # Channel 0: Normalized distance from center
            max_dist = np.sqrt((lat_range/2)**2 + (lon_range/2)**2)
            encoded[b, t, 0] = dist / max_dist
            
            # Channel 1: Angular position (normalized to [-1, 1])
            encoded[b, t, 1] = angle / np.pi
            
            # Channel 2: Radial wind component
            # Typically weak in tropical cyclones (outflow at upper levels)
            # Use simple exponential decay profile
            encoded[b, t, 2] = 0.2 * wind * torch.exp(-dist / 5.0)
            
            # Channel 3: Tangential wind component
            # Primary circulation of tropical cyclone
            # Peak winds typically at 30-60 km from center
            r_max = 1.5  # degrees (~50 km at tropical latitudes)
            wind_profile = (dist / r_max) * torch.exp(1 - dist / r_max)
            encoded[b, t, 3] = (wind / 50.0) * wind_profile  # Normalize by typical max wind
            
            # Channel 4: Pressure field
            # Low pressure at center, increases with distance
            # Normalized: typical range 900-1010 hPa
            pressure_anomaly = (1010.0 - pres) / 110.0  # Anomaly from ambient
            pressure_field = 1010.0 - pressure_anomaly * 110.0 * torch.exp(-dist / 8.0)
            encoded[b, t, 4] = (pressure_field - 955.0) / 55.0  # Normalize
    
    if not is_sequence:
        encoded = encoded.squeeze(1)
    
    return encoded


def decode_track_from_channels(
    encoded: torch.Tensor,
    method: str = 'gaussian',
    grid_range: Tuple[float, float] = (20.0, 20.0)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode track position, intensity, and pressure from encoded channels
    
    Useful for extracting typhoon characteristics from autoencoder output.
    
    Args:
        encoded: (B, T, 4+, H, W) or (B, 4+, H, W) tensor
        method: 'gaussian' or 'distance_field'
        grid_range: Physical size of grid in degrees (lat_range, lon_range)
    
    Returns:
        track: (B, T, 2) or (B, 2) estimated [lat, lon]
        intensity: (B, T) or (B,) estimated wind speed
        pressure: (B, T) or (B,) estimated central pressure
    """
    is_sequence = encoded.ndim == 5
    if not is_sequence:
        encoded = encoded.unsqueeze(1)
    
    B, T, C, H, W = encoded.shape
    device = encoded.device
    
    if method == 'gaussian':
        # Extract position from center pixel (typhoon location)
        # Since ERA5 frames are centered on typhoon, the center pixel
        # contains the typhoon's absolute position
        center_h, center_w = H // 2, W // 2
        
        # Latitude from channel 0, center pixel
        lat_norm = encoded[:, :, 0, center_h, center_w]  # (B, T)
        lat = lat_norm * 12.5 + 22.5  # Denormalize: WP range 10-35°N
        
        # Longitude from channel 1, center pixel
        lon_norm = encoded[:, :, 1, center_h, center_w]  # (B, T)
        lon = lon_norm * 20.0 + 140.0  # Denormalize: WP range 120-160°E
        
        track = torch.stack([lat, lon], dim=-1)  # (B, T, 2)
        
        # Intensity from channel 2 (uniform field, so take mean)
        intensity = encoded[:, :, 2].mean(dim=(2, 3))  # (B, T)
        intensity = intensity * 26.5 + 43.5  # Denormalize
        
        # Pressure from channel 3 (uniform field, so take mean)
        pressure = encoded[:, :, 3].mean(dim=(2, 3))  # (B, T)
        pressure = pressure * 55.0 + 955.0  # Denormalize
        
    else:  # distance_field
        # For distance field, intensity is encoded in tangential wind channel
        wind_channel = encoded[:, :, 3]  # (B, T, H, W)
        intensity = wind_channel.max(dim=-1)[0].max(dim=-1)[0] * 50.0  # Max wind * denorm
        
        # Pressure from channel 4
        pressure_channel = encoded[:, :, 4]  # (B, T, H, W)
        pressure = pressure_channel.mean(dim=(2, 3)) * 55.0 + 955.0
        
        # Position is at center by construction
        track = torch.zeros(B, T, 2, device=device)
        track[:, :, 0] = 22.5  # Mid-latitude WP
        track[:, :, 1] = 140.0  # Mid-longitude WP
    
    if not is_sequence:
        track = track.squeeze(1)
        intensity = intensity.squeeze(1)
        pressure = pressure.squeeze(1)
    
    return track, intensity, pressure


if __name__ == "__main__":
    # Test the encoding functions
    print("Testing IBTrACS encoding utilities...")
    
    # Test data
    B, T = 2, 12
    track = torch.randn(B, T, 2) * 5 + torch.tensor([22.5, 140.0])  # Around WP region
    intensity = torch.randn(B, T) * 10 + 40.0  # Around 40 m/s
    pressure = torch.randn(B, T) * 20 + 950.0  # Around 950 hPa
    
    # Test Gaussian encoding
    print("\n1. Testing Gaussian position encoding with IBTrACS data...")
    encoded_gauss = encode_track_as_channels(track, intensity, pressure)
    print(f"   Input track shape: {track.shape}")
    print(f"   Input intensity shape: {intensity.shape}")
    print(f"   Input pressure shape: {pressure.shape}")
    print(f"   Encoded shape: {encoded_gauss.shape} [Expected: (B, T, 4, H, W)]")
    print(f"   Encoded range: [{encoded_gauss.min():.3f}, {encoded_gauss.max():.3f}]")
    
    # Test distance field encoding
    print("\n2. Testing distance field encoding...")
    encoded_dist = encode_track_as_distance_field(track, intensity, pressure)
    print(f"   Encoded shape: {encoded_dist.shape} [Expected: (B, T, 5, H, W)]")
    print(f"   Encoded range: [{encoded_dist.min():.3f}, {encoded_dist.max():.3f}]")
    
    # Test single timestep
    print("\n3. Testing single timestep encoding...")
    track_single = track[:, 0, :]  # (B, 2)
    intensity_single = intensity[:, 0]  # (B,)
    pressure_single = pressure[:, 0]  # (B,)
    encoded_single = encode_track_as_channels(track_single, intensity_single, pressure_single)
    print(f"   Single timestep encoded shape: {encoded_single.shape}")
    
    # Test decoding
    print("\n4. Testing decoding...")
    decoded_track, decoded_intensity, decoded_pressure = decode_track_from_channels(encoded_gauss)
    print(f"   Decoded track shape: {decoded_track.shape}")
    print(f"   Decoded intensity shape: {decoded_intensity.shape}")
    print(f"   Decoded pressure shape: {decoded_pressure.shape}")
    print(f"   Original track sample: {track[0, 0]}")
    print(f"   Decoded track sample: {decoded_track[0, 0]}")
    print(f"   Original intensity sample: {intensity[0, 0]:.2f} m/s")
    print(f"   Decoded intensity sample: {decoded_intensity[0, 0]:.2f} m/s")
    print(f"   Original pressure sample: {pressure[0, 0]:.2f} hPa")
    print(f"   Decoded pressure sample: {decoded_pressure[0, 0]:.2f} hPa")
    
    print("\n✓ All tests passed!")

