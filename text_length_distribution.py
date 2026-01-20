#!/usr/bin/env python3
"""
Text Length Distribution Analysis - Improved Version

This script creates a text length distribution plot using KDE line plots
instead of histograms for better visibility in presentations.

Usage:
    python3 text_length_distribution.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
DATA_PATH = 'preprocessed_data_FINAL_FINAL.csv'
OUTPUT_PATH = 'plots/text_length_distribution.png'

# Set plotting style for presentations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2.5


def load_data():
    """Load the preprocessed dataset."""
    if not os.path.exists(DATA_PATH):
        # Try alternative paths
        alt_paths = [
            'combined_datasets_clean.csv',
            'combined_dataset.csv',
            'preprocessed_data_FINAL.csv'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Loading from: {alt_path}")
                return pd.read_csv(alt_path)
        raise FileNotFoundError(f"Data file not found. Tried: {DATA_PATH}")
    
    print(f"Loading from: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def calculate_outliers(data):
    """Calculate outlier statistics using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'count': len(outliers),
        'percent': len(outliers) / len(data) * 100,
        'min_outlier': outliers.min() if len(outliers) > 0 else None,
        'max_outlier': outliers.max() if len(outliers) > 0 else None
    }


def create_text_length_distribution(df):
    """Create text length distribution plot using KDE lines."""
    
    # Calculate text length if not present
    if 'text_length' not in df.columns:
        if 'text' in df.columns:
            df['text_length'] = df['text'].astype(str).apply(len)
        elif 'text_clean' in df.columns:
            df['text_length'] = df['text_clean'].astype(str).apply(len)
    
    # Use label_encoded column
    label_col = 'label_encoded' if 'label_encoded' in df.columns else 'label'
    
    # Filter data for each label
    legitimate_lengths = df[df[label_col] == 0]['text_length'].dropna()
    hoax_lengths = df[df[label_col] == 1]['text_length'].dropna()
    
    # Calculate outlier statistics
    legit_outliers = calculate_outliers(legitimate_lengths)
    hoax_outliers = calculate_outliers(hoax_lengths)
    
    print(f"\nText Length Statistics:")
    print(f"  Legitimate: n={len(legitimate_lengths):,}, mean={legitimate_lengths.mean():.1f}, std={legitimate_lengths.std():.1f}")
    print(f"  Hoax: n={len(hoax_lengths):,}, mean={hoax_lengths.mean():.1f}, std={hoax_lengths.std():.1f}")
    
    print(f"\nOutlier Statistics (IQR method):")
    print(f"  Legitimate: {legit_outliers['count']:,} outliers ({legit_outliers['percent']:.2f}%)")
    print(f"    - Range: [{legit_outliers['min_outlier']:.0f}, {legit_outliers['max_outlier']:.0f}]")
    print(f"    - IQR bounds: [{legit_outliers['lower_bound']:.1f}, {legit_outliers['upper_bound']:.1f}]")
    print(f"  Hoax: {hoax_outliers['count']:,} outliers ({hoax_outliers['percent']:.2f}%)")
    print(f"    - Range: [{hoax_outliers['min_outlier']:.0f}, {hoax_outliers['max_outlier']:.0f}]")
    print(f"    - IQR bounds: [{hoax_outliers['lower_bound']:.1f}, {hoax_outliers['upper_bound']:.1f}]")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left plot: KDE line plots ---
    # Create KDE line plots with fill for better visibility
    sns.kdeplot(
        data=legitimate_lengths, 
        ax=axes[0], 
        color='#2ecc71',  # Green
        label='Legitimate', 
        linewidth=3,
        fill=True, 
        alpha=0.35
    )
    sns.kdeplot(
        data=hoax_lengths, 
        ax=axes[0], 
        color='#e74c3c',  # Red
        label='Hoax', 
        linewidth=3,
        fill=True, 
        alpha=0.35
    )
    
    axes[0].set_xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Text Length Distribution by Label\n(Kernel Density Estimation)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    axes[0].set_xlim(0, 10000)
    axes[0].grid(True, alpha=0.3)
    
    # Add vertical lines for means
    legitimate_mean = legitimate_lengths.mean()
    hoax_mean = hoax_lengths.mean()
    axes[0].axvline(legitimate_mean, color='#27ae60', linestyle='--', linewidth=2, alpha=0.8)
    axes[0].axvline(hoax_mean, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add mean annotations
    axes[0].annotate(
        f'Mean: {legitimate_mean:.0f}', 
        xy=(legitimate_mean, axes[0].get_ylim()[1] * 0.8),
        fontsize=10, color='#27ae60', fontweight='bold'
    )
    axes[0].annotate(
        f'Mean: {hoax_mean:.0f}', 
        xy=(hoax_mean, axes[0].get_ylim()[1] * 0.6),
        fontsize=10, color='#c0392b', fontweight='bold'
    )
    
    # --- Right plot: Box plot with outlier annotations ---
    df_plot = df[['text_length', label_col]].dropna()
    df_plot['Label'] = df_plot[label_col].map({0: 'Legitimate', 1: 'Hoax'})
    
    box_colors = ['#2ecc71', '#e74c3c']
    bp = axes[1].boxplot(
        [legitimate_lengths, hoax_lengths],
        tick_labels=['Legitimate', 'Hoax'],
        patch_artist=True,
        widths=0.6,
        showfliers=True,  # Show outliers as individual points
        flierprops=dict(marker='o', markerfacecolor='orange', markersize=4, alpha=0.6)
    )
    
    # Color the box plot
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_ylabel('Text Length (characters)', fontsize=12, fontweight='bold')
    axes[1].set_title('Text Length Comparison\n(Box Plot with Outliers)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 10000)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box with outlier info
    stats_text = (
        f"Statistics:\n"
        f"────────────────────\n"
        f"Legitimate (n={len(legitimate_lengths):,}):\n"
        f"  Mean: {legitimate_lengths.mean():.1f}\n"
        f"  Median: {legitimate_lengths.median():.1f}\n"
        f"  IQR: [{legit_outliers['Q1']:.0f}, {legit_outliers['Q3']:.0f}]\n"
        f"  Outliers: {legit_outliers['count']:,} ({legit_outliers['percent']:.1f}%)\n"
        f"────────────────────\n"
        f"Hoax (n={len(hoax_lengths):,}):\n"
        f"  Mean: {hoax_lengths.mean():.1f}\n"
        f"  Median: {hoax_lengths.median():.1f}\n"
        f"  IQR: [{hoax_outliers['Q1']:.0f}, {hoax_outliers['Q3']:.0f}]\n"
        f"  Outliers: {hoax_outliers['count']:,} ({hoax_outliers['percent']:.1f}%)"
    )
    
    axes[1].text(
        0.98, 0.98, stats_text,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )
    
    # Add outlier range annotations
    axes[1].annotate(
        f'Outlier range:\n[{legit_outliers["min_outlier"]:.0f}, {legit_outliers["max_outlier"]:.0f}]',
        xy=(1, legit_outliers['max_outlier']),
        xytext=(1.15, legit_outliers['max_outlier']),
        fontsize=8,
        color='#27ae60',
        arrowprops=dict(arrowstyle='->', color='#27ae60', alpha=0.7),
        verticalalignment='center'
    )
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nPlot saved to: {OUTPUT_PATH}")
    
    # Also save as PDF for academic reports
    pdf_path = OUTPUT_PATH.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    return fig


def main():
    """Main function to run the analysis."""
    print("=" * 60)
    print("Text Length Distribution Analysis")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create the distribution plot
    fig = create_text_length_distribution(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()