"""
Script to read saved chains and create corner plots using anesthetic.
Following the integration pattern from anesthetic.md
"""
import os
import sys
import json
import numpy as np
from anesthetic import read_chains, make_2d_axes, make_1d_axes
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_corner_plot(result_dir, output_file=None, params_to_plot=None):
    """
    Create corner plot from saved chains using anesthetic.
    
    Args:
        result_dir: Directory containing the chains and metadata
        output_file: Output filename for the plot (default: corner_plot.png in result_dir)
        params_to_plot: List of parameter indices to plot (default: first 5 or all if less)
    """
    # Load metadata
    metadata_path = os.path.join(result_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load parameter names
    param_names_path = os.path.join(result_dir, 'param_names.txt')
    with open(param_names_path, 'r') as f:
        param_names = [line.strip() for line in f.readlines()]
    
    # Use anesthetic's read_chains function to properly load the nested sampling chains
    # The file is expected to be in format: [root]_dead-birth.txt
    chain_root = os.path.join(result_dir, 'chains')
    nested_samples = read_chains(chain_root, columns=param_names)
    
    # Select parameters to plot
    num_params = metadata['num_params']
    if params_to_plot is None:
        # Default: plot first 5 parameters, or all if fewer than 5
        num_to_plot = min(5, num_params)
        params_to_plot = param_names[:num_to_plot]
        
        # For fitted pulses, also include Npulse if it exists
        if metadata.get('fit_pulses', False) and num_params > 5:
            # Add the last parameter (Npulse) if not already included
            npulse_param = param_names[-1]
            if npulse_param not in params_to_plot:
                params_to_plot.append(npulse_param)
    else:
        # Convert indices to parameter names
        params_to_plot = [param_names[i] for i in params_to_plot]
    
    # Create the corner plot
    print(f"\nCreating corner plot for parameters: {params_to_plot}")
    
    # Create 2D axes using make_2d_axes as shown in anesthetic.md
    fig, axes = make_2d_axes(params_to_plot, figsize=(2*len(params_to_plot), 2*len(params_to_plot)))
    
    # Plot posterior only
    nested_samples.plot_2d(axes, alpha=0.7, color='blue')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = os.path.join(result_dir, 'corner_plot.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Corner plot saved to {output_file}")
    plt.close()
    
    # Also create individual 1D posterior plots
    print("\nCreating 1D posterior plots...")
    fig_1d, axes_1d = make_1d_axes(params_to_plot, figsize=(3*len(params_to_plot), 3))
    nested_samples.plot_1d(axes_1d)
    
    plt.tight_layout()
    
    # Save 1D plot
    output_1d = output_file.replace('.png', '_1d.png')
    plt.savefig(output_1d, dpi=150, bbox_inches='tight')
    print(f"1D posteriors saved to {output_1d}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("POSTERIOR STATISTICS")
    print("="*60)
    
    for param_name in params_to_plot:
        mean_val = nested_samples[param_name].mean()
        std_val = nested_samples[param_name].std()
        median_val = nested_samples[param_name].median()
        
        print(f"{param_name:15s}: mean={mean_val:8.3f} ± {std_val:8.3f}, median={median_val:8.3f}")
    
    # Print evidence if available
    try:
        stats = nested_samples.stats()
        print(f"\nLog Evidence: {stats['logZ']:.3f}")
        print(f"D_KL: {stats['D']:.3f}")
    except Exception as e:
        print(f"Could not compute statistics: {e}")
    
    print("="*60)


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create corner plots from saved nested sampling chains'
    )
    parser.add_argument(
        'result_dir',
        help='Directory containing chains_dead-birth.txt, param_names.txt, and metadata.json'
    )
    parser.add_argument(
        '--output',
        help='Output filename for the plot (default: corner_plot.png in result_dir)',
        default=None
    )
    parser.add_argument(
        '--params',
        nargs='+',
        type=int,
        help='Indices of parameters to plot (0-based)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Create the corner plot
    create_corner_plot(args.result_dir, args.output, args.params)


if __name__ == "__main__":
    # If no arguments provided, try to plot both test results
    if len(sys.argv) == 1:
        print("Usage: python plot_corner.py <result_dir> [--output <file>] [--params <indices>]")
        print("\nAttempting to plot existing test results...")
        
        # Try to plot fixed pulses result
        if os.path.exists('results_2pulses_fixed'):
            print("\n" + "="*60)
            print("PLOTTING: 2 PULSES FIXED")
            print("="*60)
            try:
                create_corner_plot('results_2pulses_fixed')
            except Exception as e:
                print(f"Error: {e}")
        
        # Try to plot fitted pulses result
        if os.path.exists('results_2pulses_fitted'):
            print("\n" + "="*60)
            print("PLOTTING: 2 PULSES FITTED")
            print("="*60)
            try:
                create_corner_plot('results_2pulses_fitted')
            except Exception as e:
                print(f"Error: {e}")
    else:
        main()