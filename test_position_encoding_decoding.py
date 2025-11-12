"""
Test script to verify that position encoding and decoding work correctly
with the new spatially-varying coordinate fields
"""

import torch
import numpy as np
from data.utils.ibtracs_encoding import encode_track_as_channels, decode_track_from_channels


def test_encoding_decoding():
    """Test that we can encode and then decode position accurately"""
    
    print("=" * 70)
    print("Testing IBTrACS Position Encoding and Decoding")
    print("=" * 70)
    
    # Create test data with known values
    B, T = 3, 12
    
    # Use specific positions across Western Pacific
    test_positions = [
        [15.0, 125.0],  # South-west
        [22.5, 140.0],  # Center
        [30.0, 155.0],  # North-east
    ]
    
    track = torch.tensor([
        [test_positions[i] for _ in range(T)] for i in range(B)
    ], dtype=torch.float32)  # (B, T, 2)
    
    # Add slight variations to make each timestep unique
    track = track + torch.randn_like(track) * 0.5
    
    intensity = torch.randn(B, T) * 10 + 40.0  # Around 40 m/s
    pressure = torch.randn(B, T) * 20 + 950.0  # Around 950 hPa
    
    print(f"\n1. Original Input Data:")
    print(f"   Track shape: {track.shape}")
    print(f"   Sample positions (batch 0, first 3 timesteps):")
    for t in range(3):
        print(f"     t={t}: lat={track[0, t, 0].item():.2f}°N, lon={track[0, t, 1].item():.2f}°E")
    print(f"   Sample intensity (batch 0, t=0): {intensity[0, 0].item():.2f} m/s")
    print(f"   Sample pressure (batch 0, t=0): {pressure[0, 0].item():.2f} hPa")
    
    # Encode
    print(f"\n2. Encoding to spatial channels...")
    encoded = encode_track_as_channels(
        track, 
        intensity, 
        pressure,
        image_size=(64, 64),
        normalize=True
    )
    
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Expected: (B={B}, T={T}, C=4, H=64, W=64)")
    
    # Verify spatial structure
    print(f"\n3. Verifying spatial structure of encoded channels...")
    
    # Channel 0 (latitude) - should vary vertically
    lat_channel = encoded[0, 0, 0]  # First batch, first timestep
    print(f"   Latitude channel (Ch 0):")
    print(f"     Top row (north):    {lat_channel[0, 32].item():.3f}")
    print(f"     Center row:         {lat_channel[32, 32].item():.3f}")
    print(f"     Bottom row (south): {lat_channel[63, 32].item():.3f}")
    print(f"     Expected: values decrease from top to bottom (north to south)")
    
    # Channel 1 (longitude) - should vary horizontally
    lon_channel = encoded[0, 0, 1]
    print(f"   Longitude channel (Ch 1):")
    print(f"     Left column (west):  {lon_channel[32, 0].item():.3f}")
    print(f"     Center column:       {lon_channel[32, 32].item():.3f}")
    print(f"     Right column (east): {lon_channel[32, 63].item():.3f}")
    print(f"     Expected: values increase from left to right (west to east)")
    
    # Channels 2-3 (intensity, pressure) - should be uniform
    intensity_channel = encoded[0, 0, 2]
    pressure_channel = encoded[0, 0, 3]
    print(f"   Intensity channel (Ch 2):")
    print(f"     Min: {intensity_channel.min().item():.3f}, Max: {intensity_channel.max().item():.3f}")
    print(f"     Expected: uniform (min ≈ max)")
    print(f"   Pressure channel (Ch 3):")
    print(f"     Min: {pressure_channel.min().item():.3f}, Max: {pressure_channel.max().item():.3f}")
    print(f"     Expected: uniform (min ≈ max)")
    
    # Decode
    print(f"\n4. Decoding from spatial channels...")
    decoded_track, decoded_intensity, decoded_pressure = decode_track_from_channels(
        encoded, method='gaussian'
    )
    
    print(f"   Decoded track shape: {decoded_track.shape}")
    print(f"   Decoded intensity shape: {decoded_intensity.shape}")
    print(f"   Decoded pressure shape: {decoded_pressure.shape}")
    
    # Compare original vs decoded
    print(f"\n5. Comparing Original vs Decoded:")
    print(f"\n   {'Metric':<20} {'Original':<15} {'Decoded':<15} {'Error':<15}")
    print(f"   {'-'*65}")
    
    for b in range(B):
        print(f"\n   Batch {b}:")
        for t in range(min(3, T)):  # Show first 3 timesteps
            orig_lat = track[b, t, 0].item()
            orig_lon = track[b, t, 1].item()
            orig_wind = intensity[b, t].item()
            orig_pres = pressure[b, t].item()
            
            dec_lat = decoded_track[b, t, 0].item()
            dec_lon = decoded_track[b, t, 1].item()
            dec_wind = decoded_intensity[b, t].item()
            dec_pres = decoded_pressure[b, t].item()
            
            lat_error = abs(orig_lat - dec_lat)
            lon_error = abs(orig_lon - dec_lon)
            wind_error = abs(orig_wind - dec_wind)
            pres_error = abs(orig_pres - dec_pres)
            
            print(f"     t={t}, Latitude:   {orig_lat:>7.2f}°N     {dec_lat:>7.2f}°N     {lat_error:>7.3f}°")
            print(f"     t={t}, Longitude:  {orig_lon:>7.2f}°E     {dec_lon:>7.2f}°E     {lon_error:>7.3f}°")
            print(f"     t={t}, Intensity:  {orig_wind:>7.2f} m/s   {dec_wind:>7.2f} m/s   {wind_error:>7.3f} m/s")
            print(f"     t={t}, Pressure:   {orig_pres:>7.2f} hPa   {dec_pres:>7.2f} hPa   {pres_error:>7.3f} hPa")
    
    # Calculate overall errors
    lat_mae = (track[:, :, 0] - decoded_track[:, :, 0]).abs().mean().item()
    lon_mae = (track[:, :, 1] - decoded_track[:, :, 1]).abs().mean().item()
    wind_mae = (intensity - decoded_intensity).abs().mean().item()
    pres_mae = (pressure - decoded_pressure).abs().mean().item()
    
    print(f"\n6. Mean Absolute Errors (across all batches and timesteps):")
    print(f"   Latitude:  {lat_mae:.4f}°")
    print(f"   Longitude: {lon_mae:.4f}°")
    print(f"   Intensity: {wind_mae:.4f} m/s")
    print(f"   Pressure:  {pres_mae:.4f} hPa")
    
    # Success criteria
    print(f"\n7. Validation:")
    lat_ok = lat_mae < 0.5  # Within 0.5 degrees (~55 km)
    lon_ok = lon_mae < 0.5
    wind_ok = wind_mae < 1.0  # Within 1 m/s
    pres_ok = pres_mae < 2.0  # Within 2 hPa
    
    results = [
        ("Latitude decoding", lat_ok, f"{lat_mae:.4f}° < 0.5°"),
        ("Longitude decoding", lon_ok, f"{lon_mae:.4f}° < 0.5°"),
        ("Intensity decoding", wind_ok, f"{wind_mae:.4f} < 1.0 m/s"),
        ("Pressure decoding", pres_ok, f"{pres_mae:.4f} < 2.0 hPa"),
    ]
    
    for name, passed, details in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {name:<25} ({details})")
    
    all_pass = all(r[1] for r in results)
    
    print(f"\n{'='*70}")
    if all_pass:
        print("✅ ALL TESTS PASSED! Encoding/decoding preserves position accurately.")
    else:
        print("❌ SOME TESTS FAILED! Review the errors above.")
    print(f"{'='*70}\n")
    
    return all_pass


if __name__ == "__main__":
    success = test_encoding_decoding()
    exit(0 if success else 1)

