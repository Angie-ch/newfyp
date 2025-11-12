"""
Visualize the IBTrACS encoding to understand the spatial structure
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.utils.ibtracs_encoding import encode_track_as_channels


def visualize_encoding():
    """Create visualizations of the encoded IBTrACS channels"""
    
    # Create a single test case
    track = torch.tensor([[[22.5, 140.0]]], dtype=torch.float32)  # (1, 1, 2) - center of WP
    intensity = torch.tensor([[45.0]], dtype=torch.float32)  # (1, 1) - strong typhoon
    pressure = torch.tensor([[950.0]], dtype=torch.float32)  # (1, 1) - low pressure
    
    # Encode
    encoded = encode_track_as_channels(
        track, intensity, pressure,
        image_size=(64, 64),
        normalize=True
    )  # (1, 1, 4, 64, 64)
    
    # Remove batch and time dimensions
    encoded = encoded[0, 0]  # (4, 64, 64)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('IBTrACS Encoding Visualization\n(Typhoon at 22.5°N, 140.0°E, 45 m/s, 950 hPa)', 
                 fontsize=14, fontweight='bold')
    
    channel_names = [
        'Channel 0: Latitude Field',
        'Channel 1: Longitude Field',
        'Channel 2: Intensity Field',
        'Channel 3: Pressure Field'
    ]
    
    descriptions = [
        'Vertical gradient: Top (north) to Bottom (south)\nEach row encodes its absolute latitude',
        'Horizontal gradient: Left (west) to Right (east)\nEach column encodes its absolute longitude',
        'Uniform field: Wind speed (45 m/s)\nSame value at all spatial locations',
        'Uniform field: Central pressure (950 hPa)\nSame value at all spatial locations'
    ]
    
    for idx, (ax, name, desc) in enumerate(zip(axes.flat, channel_names, descriptions)):
        data = encoded[idx].numpy()
        
        # Plot
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{name}\n{desc}', fontsize=10, pad=10)
        ax.set_xlabel('Longitude (pixel)', fontsize=9)
        ax.set_ylabel('Latitude (pixel)', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Value', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Mark center
        ax.plot(32, 32, 'r*', markersize=15, label='Typhoon Center')
        ax.legend(loc='upper right', fontsize=8)
        
        # Add value annotations at key points
        if idx == 0:  # Latitude channel
            ax.text(32, 5, f'{data[5, 32]:.2f}', ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
            ax.text(32, 32, f'{data[32, 32]:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=8)
            ax.text(32, 58, f'{data[58, 32]:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        elif idx == 1:  # Longitude channel
            ax.text(5, 32, f'{data[32, 5]:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
            ax.text(32, 32, f'{data[32, 32]:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=8)
            ax.text(58, 32, f'{data[32, 58]:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        else:  # Uniform fields
            ax.text(32, 32, f'{data[32, 32]:.2f}', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'ibtracs_encoding_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Channel Statistics:")
    print(f"{'='*70}")
    for idx, name in enumerate(channel_names):
        data = encoded[idx].numpy()
        print(f"\n{name}")
        print(f"  Min:    {data.min():.4f}")
        print(f"  Max:    {data.max():.4f}")
        print(f"  Mean:   {data.mean():.4f}")
        print(f"  Std:    {data.std():.4f}")
        print(f"  Center: {data[32, 32]:.4f}")
        
        if idx == 0:  # Latitude
            print(f"  Gradient: Top to Bottom = {data[0, 32]:.4f} → {data[63, 32]:.4f}")
        elif idx == 1:  # Longitude
            print(f"  Gradient: Left to Right = {data[32, 0]:.4f} → {data[32, 63]:.4f}")
        else:  # Uniform
            print(f"  Uniformity: All pixels = {data[32, 32]:.4f} (uniform)")
    
    print(f"\n{'='*70}")
    print("Key Insights:")
    print(f"{'='*70}")
    print("1. Latitude channel: Creates a vertical gradient")
    print("   - Decoder can read ANY row to know its latitude")
    print("   - Center pixel (32,32) tells typhoon's exact latitude")
    print()
    print("2. Longitude channel: Creates a horizontal gradient")
    print("   - Decoder can read ANY column to know its longitude")
    print("   - Center pixel (32,32) tells typhoon's exact longitude")
    print()
    print("3. Intensity & Pressure: Uniform fields")
    print("   - Accessible at any spatial location")
    print("   - Efficient for Conv layers to process")
    print()
    print("4. Benefits:")
    print("   - Position is recoverable from decoder output")
    print("   - Each pixel 'knows' its geographic coordinates")
    print("   - Spatially-connected with ERA5 atmospheric fields")
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    try:
        output = visualize_encoding()
        print(f"✅ Visualization complete! Open '{output}' to view.")
    except ImportError as e:
        print(f"⚠️  Warning: matplotlib not available. Skipping visualization.")
        print(f"   Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

