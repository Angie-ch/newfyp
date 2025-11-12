"""
Visualize the difference between random split and temporal split

This script creates a visual comparison showing:
1. Random split (WRONG) - same typhoon in train and test
2. Temporal split (CORRECT) - different typhoons in train and test
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def visualize_random_split():
    """Visualize the WRONG way - random split with data leakage"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define typhoons and their samples
    typhoons = {
        '舒力基 (2021)': {'samples': 5, 'color': '#FF6B6B'},
        '查帕卡 (2021)': {'samples': 4, 'color': '#4ECDC4'},
        '烟花 (2021)': {'samples': 3, 'color': '#45B7D1'},
        '灿都 (2021)': {'samples': 4, 'color': '#96CEB4'},
        '电母 (2022)': {'samples': 3, 'color': '#FFEAA7'},
        '梅花 (2022)': {'samples': 4, 'color': '#DFE6E9'},
    }
    
    # Create all samples
    all_samples = []
    sample_y = 0
    sample_height = 0.8
    spacing = 1.0
    
    for typhoon, info in typhoons.items():
        for i in range(info['samples']):
            all_samples.append({
                'name': f"{typhoon}_s{i}",
                'typhoon': typhoon,
                'color': info['color'],
                'y': sample_y
            })
            sample_y += spacing
    
    # Randomly assign to splits (70% train, 15% val, 15% test)
    np.random.seed(42)
    n_total = len(all_samples)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    shuffled_indices = np.random.permutation(n_total)
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train+n_val]
    test_indices = shuffled_indices[n_train+n_val:]
    
    # Draw samples
    for idx, sample in enumerate(all_samples):
        y = sample['y']
        
        # Determine split
        if idx in train_indices:
            x = 0
            split = 'Train'
            edge_color = 'green'
            edge_width = 2
        elif idx in val_indices:
            x = 5
            split = 'Val'
            edge_color = 'orange'
            edge_width = 2
        else:
            x = 10
            split = 'Test'
            edge_color = 'red'
            edge_width = 2
        
        # Draw box
        box = FancyBboxPatch(
            (x, y), 4, sample_height,
            boxstyle="round,pad=0.1",
            facecolor=sample['color'],
            edgecolor=edge_color,
            linewidth=edge_width,
            alpha=0.7
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x + 2, y + sample_height/2, sample['typhoon'],
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw arrows showing data leakage
    leakage_examples = []
    for typhoon in typhoons.keys():
        typhoon_samples = [(i, s) for i, s in enumerate(all_samples) if s['typhoon'] == typhoon]
        train_samples = [i for i, s in typhoon_samples if i in train_indices]
        test_samples = [i for i, s in typhoon_samples if i in test_indices]
        
        if train_samples and test_samples:
            leakage_examples.append((train_samples[0], test_samples[0], typhoon))
    
    # Draw leakage arrows
    for train_idx, test_idx, typhoon in leakage_examples[:3]:  # Show first 3 examples
        train_y = all_samples[train_idx]['y'] + sample_height/2
        test_y = all_samples[test_idx]['y'] + sample_height/2
        
        ax.annotate('', xy=(10, test_y), xytext=(4, train_y),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
        
        # Add warning icon
        ax.text(7, (train_y + test_y) / 2, '⚠️ 泄漏!',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Labels
    ax.text(2, -1.5, '训练集 (70%)', ha='center', fontsize=14, fontweight='bold', color='green')
    ax.text(7, -1.5, '验证集 (15%)', ha='center', fontsize=14, fontweight='bold', color='orange')
    ax.text(12, -1.5, '测试集 (15%)', ha='center', fontsize=14, fontweight='bold', color='red')
    
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, sample_y + 1)
    ax.axis('off')
    
    ax.set_title('❌ 错误的随机划分方法 - 存在数据泄漏！\n同一台风的样本出现在训练集和测试集中',
                fontsize=16, fontweight='bold', color='red', pad=20)
    
    plt.tight_layout()
    return fig


def visualize_temporal_split():
    """Visualize the CORRECT way - temporal split by year"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define typhoons by year
    typhoons_by_year = {
        2021: [
            ('舒力基', '#FF6B6B', 3),
            ('查帕卡', '#4ECDC4', 3),
            ('烟花', '#45B7D1', 2),
        ],
        2022: [
            ('灿都', '#96CEB4', 3),
            ('电母', '#FFEAA7', 2),
            ('梅花', '#DFE6E9', 3),
        ],
        2023: [
            ('泰利', '#A29BFE', 2),
            ('杜苏芮', '#FD79A8', 3),
        ],
        2024: [
            ('摩羯', '#FDCB6E', 2),
            ('贝碧嘉', '#00B894', 3),
        ]
    }
    
    # Assign splits
    year_to_split = {
        2021: ('Train', 'green', 0),
        2022: ('Train', 'green', 0),
        2023: ('Val', 'orange', 5),
        2024: ('Test', 'red', 10)
    }
    
    sample_y = 0
    sample_height = 0.8
    spacing = 1.0
    
    # Draw samples
    for year in [2021, 2022, 2023, 2024]:
        split, edge_color, x_base = year_to_split[year]
        
        for typhoon_name, color, n_samples in typhoons_by_year[year]:
            for i in range(n_samples):
                y = sample_y
                
                # Draw box
                box = FancyBboxPatch(
                    (x_base, y), 4, sample_height,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(box)
                
                # Add text
                ax.text(x_base + 2, y + sample_height/2, f"{typhoon_name} ({year})",
                       ha='center', va='center', fontsize=8, fontweight='bold')
                
                sample_y += spacing
        
        # Add year separator
        if year < 2024:
            ax.axhline(y=sample_y - 0.5, color='gray', linestyle='--', alpha=0.3)
            sample_y += 0.5
    
    # Draw checkmarks showing no leakage
    ax.text(2, sample_y + 1, '✓', ha='center', fontsize=40, color='green')
    ax.text(7, sample_y + 1, '✓', ha='center', fontsize=40, color='green')
    ax.text(12, sample_y + 1, '✓', ha='center', fontsize=40, color='green')
    
    # Labels
    ax.text(2, -1.5, '训练集\n(2021-2022)', ha='center', fontsize=14, fontweight='bold', color='green')
    ax.text(7, -1.5, '验证集\n(2023)', ha='center', fontsize=14, fontweight='bold', color='orange')
    ax.text(12, -1.5, '测试集\n(2024)', ha='center', fontsize=14, fontweight='bold', color='red')
    
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, sample_y + 2)
    ax.axis('off')
    
    ax.set_title('✅ 正确的时间划分方法 - 无数据泄漏！\n不同年份的台风分别用于训练、验证和测试',
                fontsize=16, fontweight='bold', color='green', pad=20)
    
    plt.tight_layout()
    return fig


def create_comparison_plot():
    """Create side-by-side comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Simplified versions for side-by-side
    
    # LEFT: Random split (WRONG)
    typhoons_wrong = ['舒力基', '舒力基', '舒力基', '查帕卡', '查帕卡', 
                      '烟花', '烟花', '灿都', '灿都', '电母']
    colors_wrong = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4',
                   '#45B7D1', '#45B7D1', '#96CEB4', '#96CEB4', '#FFEAA7']
    
    np.random.seed(42)
    shuffled = list(zip(typhoons_wrong, colors_wrong))
    np.random.shuffle(shuffled)
    typhoons_wrong, colors_wrong = zip(*shuffled)
    
    y_pos = np.arange(len(typhoons_wrong))
    
    # Train samples (first 7)
    ax1.barh(y_pos[:7], [1]*7, left=0, height=0.8, 
            color=[colors_wrong[i] for i in range(7)],
            edgecolor='green', linewidth=3, alpha=0.7)
    
    # Test samples (last 3)
    ax1.barh(y_pos[7:], [1]*3, left=2, height=0.8,
            color=[colors_wrong[i] for i in range(7, 10)],
            edgecolor='red', linewidth=3, alpha=0.7)
    
    # Add labels
    for i, (typhoon, color) in enumerate(zip(typhoons_wrong, colors_wrong)):
        if i < 7:
            ax1.text(0.5, i, typhoon, ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            ax1.text(2.5, i, typhoon, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Highlight data leakage
    for i in range(7):
        if typhoons_wrong[i] in typhoons_wrong[7:]:
            j = 7 + list(typhoons_wrong[7:]).index(typhoons_wrong[i])
            ax1.annotate('', xy=(2, j), xytext=(1, i),
                       arrowprops=dict(arrowstyle='->', color='red', lw=3, linestyle='--'))
    
    ax1.text(0.5, -1.5, '训练集', ha='center', fontsize=14, fontweight='bold', color='green')
    ax1.text(2.5, -1.5, '测试集', ha='center', fontsize=14, fontweight='bold', color='red')
    ax1.text(1.5, 11, '❌ 数据泄漏！', ha='center', fontsize=16, fontweight='bold', 
            color='red', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-2, 12)
    ax1.axis('off')
    ax1.set_title('错误：随机划分样本', fontsize=16, fontweight='bold', color='red', pad=20)
    
    # RIGHT: Temporal split (CORRECT)
    typhoons_2021 = ['舒力基', '舒力基', '查帕卡', '烟花']
    typhoons_2024 = ['摩羯', '摩羯', '贝碧嘉', '贝碧嘉']
    colors_2021 = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#45B7D1']
    colors_2024 = ['#FDCB6E', '#FDCB6E', '#00B894', '#00B894']
    
    y_train = np.arange(len(typhoons_2021))
    y_test = np.arange(len(typhoons_2024))
    
    # Train samples (2021-2022)
    ax2.barh(y_train, [1]*len(typhoons_2021), left=0, height=0.8,
            color=colors_2021, edgecolor='green', linewidth=3, alpha=0.7)
    
    # Test samples (2024)
    ax2.barh(y_test + 5, [1]*len(typhoons_2024), left=2, height=0.8,
            color=colors_2024, edgecolor='red', linewidth=3, alpha=0.7)
    
    # Add labels
    for i, typhoon in enumerate(typhoons_2021):
        ax2.text(0.5, i, f"{typhoon}\n(2021)", ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    for i, typhoon in enumerate(typhoons_2024):
        ax2.text(2.5, i + 5, f"{typhoon}\n(2024)", ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    ax2.text(0.5, -1.5, '训练集\n(2021-2022)', ha='center', fontsize=14, 
            fontweight='bold', color='green')
    ax2.text(2.5, -1.5, '测试集\n(2024)', ha='center', fontsize=14, 
            fontweight='bold', color='red')
    ax2.text(1.5, 11, '✅ 无数据泄漏！', ha='center', fontsize=16, fontweight='bold',
            color='green', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-2, 12)
    ax2.axis('off')
    ax2.set_title('正确：按年份划分', fontsize=16, fontweight='bold', color='green', pad=20)
    
    plt.suptitle('数据划分方法对比', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def main():
    """Create all visualizations"""
    
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("Creating visualizations...")
    
    # Create comparison plot
    fig_comparison = create_comparison_plot()
    fig_comparison.savefig('data_split_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: data_split_comparison.png")
    
    # Create detailed random split visualization
    fig_random = visualize_random_split()
    fig_random.savefig('random_split_detailed.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: random_split_detailed.png")
    
    # Create detailed temporal split visualization
    fig_temporal = visualize_temporal_split()
    fig_temporal.savefig('temporal_split_detailed.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: temporal_split_detailed.png")
    
    print("\n✅ All visualizations created!")
    print("\nFiles created:")
    print("  1. data_split_comparison.png - Side-by-side comparison")
    print("  2. random_split_detailed.png - Detailed view of random split (WRONG)")
    print("  3. temporal_split_detailed.png - Detailed view of temporal split (CORRECT)")
    
    plt.show()


if __name__ == "__main__":
    main()

