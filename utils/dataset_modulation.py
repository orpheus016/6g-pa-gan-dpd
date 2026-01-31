# filepath: utils/dataset_modulation.py
"""
Dataset Modulation Analysis
============================

This script analyzes the modulation scheme of the DPA_200MHz dataset
by plotting the I/Q constellation diagram.

For 64-QAM modulation, the constellation should show a grid-like pattern
with 64 distinct points (8x8 grid).

Dataset Specifications:
- Sample rate: 800 MSps
- Main channel BW: 200 MHz
- Subchannels: 10
- Subchannel BW: 20.0 MHz
- Expected modulation: 64-QAM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_iq_data(csv_path, max_samples=10000):
    """
    Load I/Q data from CSV file.
    
    Args:
        csv_path: Path to CSV file with I and Q columns
        max_samples: Maximum number of samples to load (for plotting efficiency)
        
    Returns:
        I, Q arrays
    """
    df = pd.read_csv(csv_path)
    
    if not {'I', 'Q'}.issubset(df.columns):
        raise ValueError("CSV must contain 'I' and 'Q' columns.")
    
    # Limit samples for plotting
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    return df['I'].values, df['Q'].values


def plot_constellation(I, Q, title="I/Q Constellation Diagram", figsize=(10, 10)):
    """
    Plot I/Q constellation diagram.
    
    For 64-QAM, this should show an 8x8 grid of points.
    
    Args:
        I: In-phase component
        Q: Quadrature component
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with transparency to see density
    ax.scatter(I, Q, alpha=0.3, s=1, c='blue', edgecolors='none')
    
    # Equal aspect ratio for proper constellation view
    ax.set_aspect('equal')
    
    # Labels and title
    ax.set_xlabel('In-phase (I)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quadrature (Q)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Add reference lines at zero
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    return fig, ax


def analyze_modulation(I, Q):
    """
    Analyze modulation scheme by counting unique constellation points.
    
    For 64-QAM, we expect approximately 64 distinct clusters.
    
    Args:
        I: In-phase component
        Q: Quadrature component
        
    Returns:
        Dictionary with analysis results
    """
    # Normalize to unit average power for analysis
    power = I**2 + Q**2
    avg_power = np.mean(power)
    I_norm = I / np.sqrt(avg_power)
    Q_norm = Q / np.sqrt(avg_power)
    
    # Round to find unique constellation points (clustering)
    # Adjust precision based on your data
    precision = 1  # decimal places
    I_rounded = np.round(I_norm, precision)
    Q_rounded = np.round(Q_norm, precision)
    
    # Find unique constellation points
    constellation_points = np.unique(np.column_stack([I_rounded, Q_rounded]), axis=0)
    
    # Calculate metrics
    peak_power = np.max(power)
    papr_linear = peak_power / avg_power
    papr_db = 10 * np.log10(papr_linear)
    
    results = {
        'num_samples': len(I),
        'unique_points': len(constellation_points),
        'avg_power': avg_power,
        'peak_power': peak_power,
        'papr_linear': papr_linear,
        'papr_db': papr_db,
        'I_min': np.min(I),
        'I_max': np.max(I),
        'Q_min': np.min(Q),
        'Q_max': np.max(Q),
    }
    
    return results


def main(csv_path, output_path=None, max_samples=10000):
    """
    Main analysis function.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save plot
        max_samples: Maximum samples to plot
    """
    print("=" * 70)
    print("Dataset Modulation Analysis: 64-QAM Verification")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {csv_path}")
    I, Q = load_iq_data(csv_path, max_samples=max_samples)
    print(f"Loaded {len(I)} samples")
    
    # Analyze modulation
    print("\nAnalyzing modulation scheme...")
    results = analyze_modulation(I, Q)
    
    # Print results
    print("\n" + "-" * 70)
    print("ANALYSIS RESULTS")
    print("-" * 70)
    print(f"Number of samples:        {results['num_samples']}")
    print(f"Unique constellation pts: {results['unique_points']}")
    print(f"Average power:            {results['avg_power']:.6f}")
    print(f"Peak power:               {results['peak_power']:.6f}")
    print(f"PAPR (linear):            {results['papr_linear']:.3f}")
    print(f"PAPR (dB):                {results['papr_db']:.2f} dB")
    print(f"I range:                  [{results['I_min']:.4f}, {results['I_max']:.4f}]")
    print(f"Q range:                  [{results['Q_min']:.4f}, {results['Q_max']:.4f}]")
    print("-" * 70)
    
    # Interpret results
    print("\nINTERPRETATION:")
    if 50 <= results['unique_points'] <= 80:
        print("✓ Constellation shows ~64 unique points → Consistent with 64-QAM")
    else:
        print(f"⚠ Constellation shows {results['unique_points']} points → May not be pure 64-QAM")
    
    if results['papr_db'] > 8:
        print(f"✓ High PAPR ({results['papr_db']:.2f} dB) → Consistent with high-order QAM")
    else:
        print(f"⚠ Moderate PAPR ({results['papr_db']:.2f} dB)")
    
    # Plot constellation
    print("\nGenerating constellation diagram...")
    dataset_name = Path(csv_path).parent.name
    fig, ax = plot_constellation(
        I, Q, 
        title=f"{dataset_name} - I/Q Constellation (64-QAM Verification)"
    )
    
    # Save or show
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Default dataset path
    default_csv = Path(__file__).parent.parent / "data" / "DPA_200MHz" / "train_input.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = str(default_csv)
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = None
    
    # Run analysis
    results = main(csv_path, output_path, max_samples=10000)
    
    print("\nTo use this script:")
    print("  python utils/dataset_modulation.py <csv_path> [output_path]")
    print("  Example: python utils/dataset_modulation.py data/DPA_200MHz/train_input.csv constellation.png")
