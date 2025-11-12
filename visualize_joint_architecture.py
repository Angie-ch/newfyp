"""
Visualization Script for Joint Encoding Architecture

Creates diagrams showing:
1. Data flow through the pipeline
2. Model architecture
3. Training/inference process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_pipeline_diagram():
    """Create complete pipeline diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Joint Encoding Pipeline for Typhoon Prediction',
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Input Stage
    input_box = FancyBboxPatch((0.5, 10), 4, 0.8, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 10.4, 'INPUT DATA (Aligned)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # ERA5 input
    era5_box = FancyBboxPatch((0.5, 9), 1.8, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor='green', facecolor='lightgreen')
    ax.add_patch(era5_box)
    ax.text(1.4, 9.3, 'ERA5\n(40,64,64)', ha='center', va='center', fontsize=8)
    
    # IBTrACS input
    ibtracs_box = FancyBboxPatch((2.7, 9), 1.8, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='orange', facecolor='lightyellow')
    ax.add_patch(ibtracs_box)
    ax.text(3.6, 9.3, 'IBTrACS\n(lat,lon,wind)', ha='center', va='center', fontsize=8)
    
    # Arrow to encoder
    arrow1 = FancyArrowPatch((2.5, 9), (2.5, 8.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Stage 1: Joint Encoding
    encoder_box = FancyBboxPatch((0.5, 6.5), 4, 1.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(2.5, 8.1, 'STAGE 1: Joint Autoencoder', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Encoder details
    ax.text(2.5, 7.6, '1. Embed IBTrACS → Spatial (16,64,64)', ha='center', fontsize=8)
    ax.text(2.5, 7.3, '2. Concat with ERA5 → (56,64,64)', ha='center', fontsize=8)
    ax.text(2.5, 7.0, '3. CNN Encoder → Latent (8,8,8)', ha='center', fontsize=8)
    ax.text(2.5, 6.7, '✓ Unified representation!', ha='center', fontsize=8,
            style='italic', color='green')
    
    # Arrow to diffusion
    arrow2 = FancyArrowPatch((2.5, 6.5), (2.5, 6.0),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Stage 2: Diffusion
    diffusion_box = FancyBboxPatch((0.5, 4.2), 4, 1.6,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='red', facecolor='mistyrose', linewidth=2)
    ax.add_patch(diffusion_box)
    ax.text(2.5, 5.6, 'STAGE 2: Diffusion Model', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Diffusion details
    ax.text(2.5, 5.2, '1. Condition on past latents', ha='center', fontsize=8)
    ax.text(2.5, 4.9, '2. Denoise future latents (T steps)', ha='center', fontsize=8)
    ax.text(2.5, 4.6, '3. Physics-informed constraints', ha='center', fontsize=8)
    ax.text(2.5, 4.3, '✓ Operates on unified space!', ha='center', fontsize=8,
            style='italic', color='green')
    
    # Arrow to decoder
    arrow3 = FancyArrowPatch((2.5, 4.2), (2.5, 3.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Stage 3: Separate Decoding
    decoder_box = FancyBboxPatch((0.5, 1.8), 4, 1.7,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='teal', facecolor='lightcyan', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(2.5, 3.3, 'STAGE 3: Separate Decoder', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Decoder details
    ax.text(2.5, 2.9, '1. CNN Decoder → Shared features', ha='center', fontsize=8)
    ax.text(2.5, 2.6, '2. ERA5 Head (Conv) → (40,64,64)', ha='center', fontsize=8)
    ax.text(2.5, 2.3, '3. IBTrACS Head (MLP) → (lat,lon,wind)', ha='center', fontsize=8)
    ax.text(2.5, 2.0, '✓ Separate outputs!', ha='center', fontsize=8,
            style='italic', color='green')
    
    # Arrow to output
    arrow4 = FancyArrowPatch((2.5, 1.8), (2.5, 1.3),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Output Stage
    output_box = FancyBboxPatch((0.5, 0.3), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(output_box)
    ax.text(2.5, 0.7, 'OUTPUT: Predictions + Uncertainty', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    # Right side: Key Benefits
    benefits_box = FancyBboxPatch((5.5, 6), 4, 5,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='gold', facecolor='lightyellow', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(7.5, 10.7, 'Key Benefits', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    benefits = [
        '✓ Unified Representation',
        '  ERA5 + IBTrACS encoded together',
        '',
        '✓ Better Information Fusion',
        '  Model learns joint correlations',
        '',
        '✓ Consistent Predictions',
        '  Single latent ensures consistency',
        '',
        '✓ Simpler Conditioning',
        '  Only past latents needed',
        '',
        '✓ End-to-End Learning',
        '  Joint optimization of all components'
    ]
    
    y_pos = 10.2
    for benefit in benefits:
        if benefit == '':
            y_pos -= 0.2
        else:
            ax.text(5.7, y_pos, benefit, ha='left', va='top', fontsize=8)
            y_pos -= 0.3
    
    # Right side: Model Stats
    stats_box = FancyBboxPatch((5.5, 0.3), 4, 5.3,
                               boxstyle="round,pad=0.1",
                               edgecolor='gray', facecolor='whitesmoke', linewidth=2)
    ax.add_patch(stats_box)
    ax.text(7.5, 5.3, 'Model Specifications', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    stats = [
        'Joint Autoencoder:',
        '  • Input: ERA5 (40,64,64) + IBTrACS (3)',
        '  • Latent: (8,8,8) - 320× compression',
        '  • Parameters: ~50M',
        '  • Training: ~6 hours (A100)',
        '',
        'Diffusion Model:',
        '  • Architecture: 3D UNet + Spiral Attn',
        '  • Hidden dim: 256',
        '  • Parameters: ~100M',
        '  • Training: ~16 hours (A100)',
        '',
        'Total Pipeline:',
        '  • Parameters: ~150M',
        '  • Training: ~22 hours (A100)',
        '  • Inference: ~2-5 sec/sample (DDIM)',
        '',
        'Expected Performance:',
        '  • 24h Track Error: <100 km',
        '  • 48h Track Error: <200 km',
        '  • Intensity MAE: <5 m/s'
    ]
    
    y_pos = 5.0
    for stat in stats:
        if stat == '':
            y_pos -= 0.15
        elif stat.endswith(':'):
            ax.text(5.7, y_pos, stat, ha='left', va='top', fontsize=8, fontweight='bold')
            y_pos -= 0.25
        else:
            ax.text(5.7, y_pos, stat, ha='left', va='top', fontsize=7)
            y_pos -= 0.25
    
    plt.tight_layout()
    return fig


def create_comparison_diagram():
    """Create comparison between separate and joint encoding"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Separate Encoding (OLD)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('OLD: Separate Encoding', fontsize=14, fontweight='bold', pad=20)
    
    # ERA5 path
    era5_1 = FancyBboxPatch((1, 8), 2, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(era5_1)
    ax1.text(2, 8.4, 'ERA5\n(40,64,64)', ha='center', va='center', fontsize=9)
    
    arrow_e1 = FancyArrowPatch((2, 8), (2, 7),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax1.add_patch(arrow_e1)
    
    ae_1 = FancyBboxPatch((1, 6), 2, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='purple', facecolor='lavender', linewidth=2)
    ax1.add_patch(ae_1)
    ax1.text(2, 6.4, 'Autoencoder\nERA5 only', ha='center', va='center', fontsize=8)
    
    arrow_e2 = FancyArrowPatch((2, 6), (2, 5),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax1.add_patch(arrow_e2)
    
    latent_1 = FancyBboxPatch((1, 4), 2, 0.8, boxstyle="round,pad=0.05",
                              edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(latent_1)
    ax1.text(2, 4.4, 'Latent\n(8,8,8)', ha='center', va='center', fontsize=9)
    
    # IBTrACS path
    ibtracs_1 = FancyBboxPatch((5, 8), 2, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax1.add_patch(ibtracs_1)
    ax1.text(6, 8.4, 'IBTrACS\n(track, int)', ha='center', va='center', fontsize=9)
    
    arrow_i1 = FancyArrowPatch((6, 8), (6, 5),
                              arrowstyle='->', mutation_scale=15, linewidth=2, linestyle='dashed')
    ax1.add_patch(arrow_i1)
    ax1.text(6.5, 6.5, 'Pass\nthrough', ha='left', va='center', fontsize=7, style='italic')
    
    ibtracs_2 = FancyBboxPatch((5, 4), 2, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax1.add_patch(ibtracs_2)
    ax1.text(6, 4.4, 'IBTrACS\n(scalars)', ha='center', va='center', fontsize=9)
    
    # Merge
    arrow_m1 = FancyArrowPatch((3, 4.4), (4, 3),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax1.add_patch(arrow_m1)
    arrow_m2 = FancyArrowPatch((5, 4.4), (4, 3),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax1.add_patch(arrow_m2)
    
    diffusion_1 = FancyBboxPatch((2.5, 1.5), 3, 1.2, boxstyle="round,pad=0.05",
                                 edgecolor='red', facecolor='mistyrose', linewidth=2)
    ax1.add_patch(diffusion_1)
    ax1.text(4, 2.1, 'Diffusion Model\n(separate inputs)', ha='center', va='center', fontsize=9)
    
    # Problems
    ax1.text(5, 1, '❌ Separate processing', ha='center', fontsize=8, color='red')
    ax1.text(5, 0.5, '❌ No joint learning', ha='center', fontsize=8, color='red')
    
    # Right: Joint Encoding (NEW)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('NEW: Joint Encoding', fontsize=14, fontweight='bold', pad=20)
    
    # ERA5 input
    era5_2 = FancyBboxPatch((1, 8), 2, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax2.add_patch(era5_2)
    ax2.text(2, 8.4, 'ERA5\n(40,64,64)', ha='center', va='center', fontsize=9)
    
    # IBTrACS input
    ibtracs_3 = FancyBboxPatch((5, 8), 2, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax2.add_patch(ibtracs_3)
    ax2.text(6, 8.4, 'IBTrACS\n(track, int)', ha='center', va='center', fontsize=9)
    
    # Merge early
    arrow_j1 = FancyArrowPatch((3, 8.4), (4, 7),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax2.add_patch(arrow_j1)
    arrow_j2 = FancyArrowPatch((5, 8.4), (4, 7),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax2.add_patch(arrow_j2)
    
    # Joint encoder
    joint_enc = FancyBboxPatch((2.5, 6), 3, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='purple', facecolor='lavender', linewidth=2)
    ax2.add_patch(joint_enc)
    ax2.text(4, 6.4, 'Joint Encoder\n(ERA5 + IBTrACS)', ha='center', va='center', fontsize=9)
    
    arrow_j3 = FancyArrowPatch((4, 6), (4, 5),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax2.add_patch(arrow_j3)
    
    # Unified latent
    unified_latent = FancyBboxPatch((2.5, 4), 3, 0.8, boxstyle="round,pad=0.05",
                                    edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax2.add_patch(unified_latent)
    ax2.text(4, 4.4, 'Unified Latent\n(8,8,8)', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(4, 3.7, 'Contains ERA5 + IBTrACS!', ha='center', va='center', fontsize=7, style='italic')
    
    arrow_j4 = FancyArrowPatch((4, 4), (4, 3),
                              arrowstyle='->', mutation_scale=15, linewidth=2)
    ax2.add_patch(arrow_j4)
    
    # Diffusion
    diffusion_2 = FancyBboxPatch((2.5, 1.5), 3, 1.2, boxstyle="round,pad=0.05",
                                 edgecolor='red', facecolor='mistyrose', linewidth=2)
    ax2.add_patch(diffusion_2)
    ax2.text(4, 2.1, 'Diffusion Model\n(unified input)', ha='center', va='center', fontsize=9)
    
    # Benefits
    ax2.text(5, 1, '✓ Joint processing', ha='center', fontsize=8, color='green')
    ax2.text(5, 0.5, '✓ Unified learning', ha='center', fontsize=8, color='green')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Create pipeline diagram
    print("Creating pipeline diagram...")
    fig1 = create_pipeline_diagram()
    fig1.savefig('joint_encoding_pipeline.png', dpi=300, bbox_inches='tight')
    print("Saved: joint_encoding_pipeline.png")
    
    # Create comparison diagram
    print("Creating comparison diagram...")
    fig2 = create_comparison_diagram()
    fig2.savefig('encoding_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: encoding_comparison.png")
    
    print("\nDiagrams created successfully!")
    print("View them to understand the joint encoding architecture.")

