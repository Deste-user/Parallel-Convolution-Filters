#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def setup_environment():
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10
    
    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def load_and_preprocess_data():
    print("Loading performance data...")
    try:
        seq_layout = pd.read_csv("experiments_results/performance_layout_seq.csv")
        cuda_layout = pd.read_csv("experiments_results/performance_layout.csv")
        cuda_tiling = pd.read_csv("experiments_results/performance_tiling.csv")
        cuda_dim_block = pd.read_csv("experiments_results/performance_dim_block.csv")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you are running the script from the correct directory.")
        exit(1)

    print(" Data loaded successfully")
    print(f"  Sequential data points: {len(seq_layout)}")
    print(f"  CUDA data points: {len(cuda_layout)}\n")

    common_factors = sorted(list(set(seq_layout['Factor'].values) & set(cuda_layout['Factor'].values)))
    seq_layout = seq_layout[seq_layout['Factor'].isin(common_factors)].reset_index(drop=True)
    cuda_layout = cuda_layout[cuda_layout['Factor'].isin(common_factors)].reset_index(drop=True)

    print(f"  Common factors: {common_factors}")
    print(f"  Working with {len(seq_layout)} common data points\n")
    
    return seq_layout, cuda_layout, cuda_tiling, cuda_dim_block

def plot_sequential_vs_cuda(seq_df, cuda_df, output_dir):
    print("Generating Figure 1: Sequential vs CUDA Layout Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    factor_array = seq_df['Factor'].values

    # AoS Comparison
    ax = axes[0]
    ax.plot(factor_array, seq_df['AOS time'], 'o-', label='Sequential AoS', linewidth=2, markersize=8)
    ax.plot(factor_array, cuda_df['AOS time'], 's-', label='CUDA AoS', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Resize Factor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('AoS Layout: Sequential vs CUDA', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(factor_array)              
    ax.set_xticklabels(factor_array)

    # SoA Comparison
    ax = axes[1]
    ax.plot(seq_df['Factor'], seq_df['SOA time'], 'o-', label='Sequential SoA', linewidth=2, markersize=8)
    ax.plot(cuda_df['Factor'], cuda_df['SOA time'], 's-', label='CUDA SoA', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Resize Factor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('SoA Layout: Sequential vs CUDA', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(factor_array)               
    ax.set_xticklabels(factor_array)

    plt.tight_layout()
    filename = "01_sequential_vs_cuda_comparison.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}\n")
    plt.close()

def plot_speedup_analysis(seq_df, cuda_df, output_dir):
    """Genera la Figura 2: Analisi dello Speedup."""
    print("Generating Figure 2: GPU Speedup Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    speedup_aos = seq_df['AOS time'] / cuda_df['AOS time']
    speedup_soa = seq_df['SOA time'] / cuda_df['SOA time']
    factors = range(len(seq_df['Factor']))

    # Funzione helper per annotare le barre
    def annotate_bars(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}x', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # AoS Speedup
    ax = axes[0]
    bars1 = ax.bar(factors, speedup_aos, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_title('AoS Speedup: Sequential vs CUDA', fontsize=12, fontweight='bold')
    annotate_bars(ax, bars1)

    # SoA Speedup
    ax = axes[1]
    bars2 = ax.bar(factors, speedup_soa, alpha=0.7, color='forestgreen', edgecolor='black')
    ax.set_title('SoA Speedup: Sequential vs CUDA', fontsize=12, fontweight='bold')
    annotate_bars(ax, bars2)


    for ax in axes:
        ax.set_xlabel('Resize Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speedup (Sequential / CUDA)', fontsize=11, fontweight='bold')
        ax.set_xticks(factors)
        ax.set_xticklabels(seq_df['Factor'])

        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    plt.tight_layout()
    filename = "02_speedup_analysis.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}\n")
    plt.close()
    return speedup_aos, speedup_soa 

def plot_layout_comparison_cuda(cuda_df, output_dir):
    print("Generating Figure 3: AoS vs SoA on GPU...")
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cuda_df['Factor']))
    width = 0.35

    ax.bar(x - width/2, cuda_df['AOS time'], width, label='CUDA AoS', alpha=0.8, color='steelblue', edgecolor='black')
    ax.bar(x + width/2, cuda_df['SOA time'], width, label='CUDA SoA', alpha=0.8, color='darkorange', edgecolor='black')

    ax.set_xlabel('Resize Factor', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Data Layout Impact on GPU Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cuda_df['Factor'])
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for i in range(len(cuda_df['Factor'])):
        ratio = cuda_df['SOA time'].iloc[i] / cuda_df['AOS time'].iloc[i]
        y_pos = max(cuda_df['AOS time'].iloc[i], cuda_df['SOA time'].iloc[i]) * 1.3
        ax.text(i, y_pos, f'{ratio:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    filename = "03_layout_comparison_cuda.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}\n")
    plt.close()

def plot_tiling_effect(cuda_tiling, output_dir):
    """Genera la Figura 4: Effetto del Tiling."""
    print("Generating Figure 4: Tiling Effect Analysis...")
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cuda_tiling['Size']))
    width = 0.35

    ax.bar(x - width/2, cuda_tiling['SOA Tiled Time'], width, label='Tiled', alpha=0.8, color='forestgreen', edgecolor='black')
    ax.bar(x + width/2, cuda_tiling['SOA Notiled Time'], width, label='Non-Tiled', alpha=0.8, color='salmon', edgecolor='black')

    ax.set_xlabel('Filter Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Tiling Optimization Impact (SoA Layout)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cuda_tiling['Size'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for i in range(len(cuda_tiling['Size'])):
        improvement = (1 - cuda_tiling['SOA Tiled Time'].iloc[i] / cuda_tiling['SOA Notiled Time'].iloc[i]) * 100
        y_pos = max(cuda_tiling['SOA Tiled Time'].iloc[i], cuda_tiling['SOA Notiled Time'].iloc[i]) * 1
        ax.text(i, y_pos + 1, f'{improvement:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    filename = "04_tiling_effect.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}\n")
    plt.close()

def plot_block_dimension_optimization(cuda_dim_block, output_dir):
    print("Generating Figure 5: Block Dimension Optimization...")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cuda_dim_block['Dim Block'], cuda_dim_block['SOA Tiled Time'], 'o-', linewidth=2.5, markersize=10, color='darkblue')
    ax.fill_between(cuda_dim_block['Dim Block'], cuda_dim_block['SOA Tiled Time'], alpha=0.3, color='lightblue')

    ax.set_xlabel('Block Dimension (DxD threads)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('GPU Block Dimension Optimization (Tiled SoA)', fontsize=12, fontweight='bold')
    ax.set_xticks(cuda_dim_block['Dim Block'])
    ax.grid(True, alpha=0.3)

    optimal_idx = cuda_dim_block['SOA Tiled Time'].idxmin()
    optimal_dim = cuda_dim_block.loc[optimal_idx, 'Dim Block']
    optimal_time = cuda_dim_block.loc[optimal_idx, 'SOA Tiled Time']
    
    ax.plot(optimal_dim, optimal_time, 'r*', markersize=20, 
            label=f'Optimal: {int(optimal_dim)}x{int(optimal_dim)} ({optimal_time:.2f}ms)')
    ax.legend(fontsize=10)

    plt.tight_layout()
    filename = "05_block_dimension_optimization.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f" Saved: {filename}\n")
    plt.close()


def print_performance_summary(seq_df, cuda_df, cuda_tiling, cuda_dim_block, speedup_aos, speedup_soa, output_dir):
    """Stampa a schermo il riepilogo delle prestazioni."""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*70)

    print("\n SEQUENTIAL vs CUDA SPEEDUP:")
    print("-" * 70)
    print(f"  AoS Layout:")
    print(f"    Average Speedup: {speedup_aos.mean():.2f}x")
    print(f"    Max Speedup:     {speedup_aos.max():.2f}x at Factor {seq_df.loc[speedup_aos.idxmax(), 'Factor']:.0f}")
    print(f"    Min Speedup:     {speedup_aos.min():.2f}x at Factor {seq_df.loc[speedup_aos.idxmin(), 'Factor']:.0f}")

    print(f"\n  SoA Layout:")
    print(f"    Average Speedup: {speedup_soa.mean():.2f}x")
    print(f"    Max Speedup:     {speedup_soa.max():.2f}x at Factor {seq_df.loc[speedup_soa.idxmax(), 'Factor']:.0f}")
    print(f"    Min Speedup:     {speedup_soa.min():.2f}x at Factor {seq_df.loc[speedup_soa.idxmin(), 'Factor']:.0f}")

    print("\n LAYOUT COMPARISON (GPU Performance):")
    print("-" * 70)
    layout_ratio = cuda_df['SOA time'] / cuda_df['AOS time']
    print(f"  SoA vs AoS on GPU:")
    print(f"    Average Ratio:   {layout_ratio.mean():.2f}x")
    print(f"    Max Ratio:       {layout_ratio.max():.2f}x (SoA {('slower' if layout_ratio.max() > 1 else 'faster')} at Factor {cuda_df.loc[layout_ratio.idxmax(), 'Factor']:.0f})")
    print(f"    Min Ratio:       {layout_ratio.min():.2f}x (SoA {('slower' if layout_ratio.min() > 1 else 'faster')} at Factor {cuda_df.loc[layout_ratio.idxmin(), 'Factor']:.0f})")

    print("\n TILING OPTIMIZATION:")
    print("-" * 70)
    tiling_gain = ((cuda_tiling['SOA Notiled Time'] - cuda_tiling['SOA Tiled Time']) / cuda_tiling['SOA Notiled Time'] * 100)
    print(f"  Average Improvement: {tiling_gain.mean():.2f}%")
    print(f"  Max Improvement:     {tiling_gain.max():.2f}% (Filter Size {cuda_tiling.loc[tiling_gain.idxmax(), 'Size']:.0f}x{int(cuda_tiling.loc[tiling_gain.idxmax(), 'Size'])})")
    print(f"  Min Improvement:     {tiling_gain.min():.2f}% (Filter Size {cuda_tiling.loc[tiling_gain.idxmin(), 'Size']:.0f}x{int(cuda_tiling.loc[tiling_gain.idxmin(), 'Size'])})")

    print("\n  BLOCK DIMENSION OPTIMIZATION:")
    print("-" * 70)
    optimal_block = cuda_dim_block.loc[cuda_dim_block['SOA Tiled Time'].idxmin()]
    worst_block = cuda_dim_block.loc[cuda_dim_block['SOA Tiled Time'].idxmax()]
    print(f"  Optimal Block Size:  {int(optimal_block['Dim Block'])}x{int(optimal_block['Dim Block'])} ({optimal_block['SOA Tiled Time']:.4f} ms)")
    print(f"  Worst Block Size:    {int(worst_block['Dim Block'])}x{int(worst_block['Dim Block'])} ({worst_block['SOA Tiled Time']:.4f} ms)")
    print(f"  Improvement:         {(worst_block['SOA Tiled Time'] / optimal_block['SOA Tiled Time'] - 1) * 100:.1f}%")

    print("\n" + "="*70)
    print(f" All plots saved to: {output_dir.absolute()}/")
    print("="*70 + "\n")

def main():
    output_dir = setup_environment()
    seq_df, cuda_df, cuda_tiling, cuda_dim_block = load_and_preprocess_data()
    

    plot_sequential_vs_cuda(seq_df, cuda_df, output_dir)
    speedup_aos, speedup_soa = plot_speedup_analysis(seq_df, cuda_df, output_dir)
    plot_layout_comparison_cuda(cuda_df, output_dir)
    plot_tiling_effect(cuda_tiling, output_dir)
    plot_block_dimension_optimization(cuda_dim_block, output_dir)
    
    print_performance_summary(seq_df, cuda_df, cuda_tiling, cuda_dim_block, speedup_aos, speedup_soa, output_dir)

if __name__ == "__main__":
    main()