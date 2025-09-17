"""
Analysis and visualization utilities for FRBayes JAX.
"""
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains, make_2d_axes, make_1d_axes, NestedSamples, read_csv as anesthetic_read_csv
from fgivenx import plot_contours, plot_lines
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
import os
from .models import get_model_function, get_param_names, get_sigma_index, get_npulse_index


# Model color schemes
MODEL_COLOR_MAPS = {
    "emg": plt.cm.Blues_r,
    "exponential": plt.cm.Greens_r,
    "emg_with_baseline": plt.cm.GnBu_r,
    "exponential_with_baseline": plt.cm.Oranges_r,
}

MODEL_COLORS = {
    "emg": "blue",
    "exponential": "green",
    "emg_with_baseline": "cyan",
    "exponential_with_baseline": "orange",
}


def plot_corner(
    chain_file: str,
    param_names: List[str],
    model_name: str,
    output_dir: str = ".",
    params_to_plot: Optional[List[int]] = None
) -> None:
    """
    Create corner plot using anesthetic.

    Args:
        chain_file: Path to chain file (without _dead-birth.txt)
        param_names: List of parameter names
        model_name: Model name for coloring
        output_dir: Directory to save plots
        params_to_plot: Indices of parameters to plot (None = all)
    """
    # Load chains - handle both old format and CSV
    if chain_file.endswith('.csv'):
        chains = anesthetic_read_csv(chain_file)
    else:
        chains = read_chains(chain_file, columns=param_names)
    
    # Remove any NaN or Inf values
    import numpy as np
    # Select only numeric columns for inf check
    numeric_chains = chains.select_dtypes(include=[np.number])
    mask = ~(numeric_chains.isnull().any(axis=1) | np.isinf(numeric_chains).any(axis=1))
    chains = chains[mask]
    
    if len(chains) == 0:
        print("Warning: No valid samples after removing NaN/Inf values")
        return
    
    # Select parameters to plot
    if params_to_plot is not None:
        # When loading from CSV, columns are multi-index tuples
        if chain_file.endswith('.csv'):
            # Get the actual column names from chains
            all_cols = [col for col in chains.columns if not isinstance(col, tuple) or col[0] not in ['logL', 'logL_birth', 'nlive']]
            if not all_cols:  # If all columns are tuples
                all_cols = [col for col in chains.columns if isinstance(col, tuple) and col[0] not in ['logL', 'logL_birth', 'nlive']]
            params_subset = [all_cols[i] for i in params_to_plot if i < len(all_cols)]
        else:
            params_subset = [param_names[i] for i in params_to_plot]
    else:
        if chain_file.endswith('.csv'):
            # Filter out non-parameter columns
            params_subset = [col for col in chains.columns if isinstance(col, tuple) and col[0] not in ['logL', 'logL_birth', 'nlive']][:12]
        else:
            params_subset = param_names

    # Create corner plot
    fig, axes = make_2d_axes(params_subset[:min(12, len(params_subset))], figsize=(10, 10), facecolor='w')
    
    # Plot with model-specific color
    color = MODEL_COLORS.get(model_name, "black")
    chains.plot_2d(axes, alpha=0.9, color=color)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "corner_plot.png")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Corner plot saved to {output_file}")


def plot_functional_posterior(
    chain_file: str,
    param_names: List[str],
    model_name: str,
    max_peaks: int,
    fit_pulses: bool,
    t: np.ndarray,
    data: Optional[np.ndarray] = None,
    output_dir: str = ".",
    nsamples: int = 500
) -> None:
    """
    Create functional posterior plot using fgivenx.
    
    Args:
        chain_file: Path to chain file (without _dead-birth.txt)
        param_names: List of parameter names
        model_name: Model name
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is fitted
        t: Time axis
        data: Optional observed data to overlay
        output_dir: Directory to save plots
        nsamples: Number of samples to use for plotting
    """
    # Load chains - handle both old format and CSV
    if chain_file.endswith('.csv'):
        chains = anesthetic_read_csv(chain_file)
    else:
        chains = read_chains(chain_file, columns=param_names)
    
    # Remove any NaN or Inf values
    import numpy as np
    # Select only numeric columns for inf check
    numeric_chains = chains.select_dtypes(include=[np.number])
    mask = ~(numeric_chains.isnull().any(axis=1) | np.isinf(numeric_chains).any(axis=1))
    chains = chains[mask]
    
    if len(chains) == 0:
        print("Warning: No valid samples after removing NaN/Inf values")
        return
    
    # Get model function
    model_func = get_model_function(model_name)
    
    # Define wrapper for fgivenx
    def model_wrapper(t_val, theta):
        """Wrapper to make model compatible with fgivenx."""
        t_array = jnp.array(t_val)
        theta_array = jnp.array(theta)
        return np.array(model_func(t_array, theta_array, max_peaks, fit_pulses))
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Get model-specific colors
    cmap = MODEL_COLOR_MAPS.get(model_name, plt.cm.Blues_r)
    color = MODEL_COLORS.get(model_name, "blue")
    
    # Plot contours
    print("Plotting functional posterior contours...")
    plot_contours(
        model_wrapper,
        t,
        chains,
        axes[0],
        weights=chains.get_weights(),
        colors=cmap,
        cache=f"{output_dir}/cache_contours"
    )
    
    # Overlay data if provided
    if data is not None:
        axes[0].plot(t, data, 'k.', alpha=0.5, markersize=2, label='Data')
        axes[0].legend()
    
    axes[0].set_ylabel('Signal', fontsize=12)
    axes[0].set_title('Functional Posterior', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot lines (individual realizations)
    print("Plotting individual realizations...")
    plot_lines(
        model_wrapper,
        t,
        chains,
        axes[1],
        weights=chains.get_weights(),
        color=color,
        alpha=0.1,
        cache=f"{output_dir}/cache_lines"
    )
    
    # Overlay data if provided
    if data is not None:
        axes[1].plot(t, data, 'k.', alpha=0.5, markersize=2, label='Data')
        axes[1].legend()
    
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Signal', fontsize=12)
    axes[1].set_title('Model Realizations', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "functional_posterior.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Functional posterior saved to {output_file}")


def plot_parameter_distributions(
    chain_file: str,
    param_names: List[str],
    model_name: str,
    max_peaks: int,
    fit_pulses: bool,
    output_dir: str = "."
) -> None:
    """
    Plot 1D parameter distributions.
    
    Args:
        chain_file: Path to chain file (without _dead-birth.txt)
        param_names: List of parameter names
        model_name: Model name
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is fitted
        output_dir: Directory to save plots
    """
    # Load chains - handle both old format and CSV
    if chain_file.endswith('.csv'):
        chains = anesthetic_read_csv(chain_file)
    else:
        chains = read_chains(chain_file, columns=param_names)
    
    # Remove any NaN or Inf values
    import numpy as np
    # Select only numeric columns for inf check
    numeric_chains = chains.select_dtypes(include=[np.number])
    mask = ~(numeric_chains.isnull().any(axis=1) | np.isinf(numeric_chains).any(axis=1))
    chains = chains[mask]
    
    if len(chains) == 0:
        print("Warning: No valid samples after removing NaN/Inf values")
        return
    
    # Get model color
    color = MODEL_COLORS.get(model_name, "blue")
    
    # Create different plots for different parameter groups
    
    # 1. Amplitude parameters
    amp_params = [p for p in param_names if 'A_' in p]
    if amp_params:
        fig, axes = make_1d_axes(amp_params, figsize=(12, 3))
        chains.plot_1d(axes, color=color)
        fig.suptitle('Amplitude Parameters', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "params_amplitude.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Tau parameters
    tau_params = [p for p in param_names if r'\tau' in p]
    if tau_params:
        fig, axes = make_1d_axes(tau_params, figsize=(12, 3))
        chains.plot_1d(axes, color=color)
        fig.suptitle('Decay Time Parameters', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "params_tau.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Arrival time parameters
    u_params = [p for p in param_names if 'u_' in p]
    if u_params:
        fig, axes = make_1d_axes(u_params, figsize=(12, 3))
        chains.plot_1d(axes, color=color)
        fig.suptitle('Arrival Time Parameters', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "params_arrival.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Width parameters (for EMG)
    w_params = [p for p in param_names if 'w_' in p]
    if w_params:
        fig, axes = make_1d_axes(w_params, figsize=(12, 3))
        chains.plot_1d(axes, color=color)
        fig.suptitle('Width Parameters', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "params_width.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Npulse if fitted
    if fit_pulses:
        npulse_param = [p for p in param_names if 'N_' in p and 'pulse' in p]
        if npulse_param:
            fig, ax = plt.subplots(figsize=(6, 4))
            npulse_chain = chains[npulse_param[0]]
            
            # Create histogram
            counts, bins, _ = ax.hist(npulse_chain, bins=np.arange(0.5, max_peaks + 1.5, 1),
                                     weights=chains.get_weights(), color=color, alpha=0.7,
                                     edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Number of Pulses', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title('Pulse Number Distribution', fontsize=14)
            ax.set_xticks(range(1, max_peaks + 1))
            ax.grid(True, alpha=0.3)
            
            # Add mean and mode
            mean_npulse = float(np.average(npulse_chain, weights=chains.get_weights()))
            mode_npulse = bins[np.argmax(counts)]
            ax.axvline(mean_npulse, color='red', linestyle='--', label=f'Mean: {mean_npulse:.2f}')
            ax.axvline(mode_npulse, color='green', linestyle='--', label=f'Mode: {int(mode_npulse)}')
            ax.legend()
            
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "npulse_distribution.png"), dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Parameter distributions saved to {output_dir}")


def analyze_results(
    results: Dict,
    model_name: str,
    t: np.ndarray,
    data: Optional[np.ndarray] = None,
    output_dir: str = "results",
    chain_file: str = "chains/chains"
) -> None:
    """
    Complete analysis pipeline for nested sampling results.
    
    Args:
        results: Sampling results dictionary
        model_name: Model name
        t: Time axis
        data: Optional observed data
        output_dir: Output directory for plots
        chain_file: Base name for chain files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter names
    param_names = get_param_names(model_name, results['max_peaks'], results['fit_pulses'])
    
    # Save chains in anesthetic format
    from .sampling import save_chains_for_anesthetic
    save_chains_for_anesthetic(results, chain_file)
    print(f"Chains saved to {chain_file}_dead-birth.txt")
    
    # Create corner plot
    print("\nCreating corner plot...")
    plot_corner(chain_file, param_names, model_name, output_dir)
    
    # Create functional posterior
    print("\nCreating functional posterior plot...")
    plot_functional_posterior(
        chain_file, param_names, model_name,
        results['max_peaks'], results['fit_pulses'],
        t, data, output_dir
    )
    
    # Create parameter distributions
    print("\nCreating parameter distribution plots...")
    plot_parameter_distributions(
        chain_file, param_names, model_name,
        results['max_peaks'], results['fit_pulses'],
        output_dir
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Max peaks: {results['max_peaks']}")
    print(f"Fit pulses: {results['fit_pulses']}")
    print(f"Log evidence: {results['logZ']:.4f} ± {results['logZ_error']:.4f}")
    print(f"Number of samples: {results['nsamples']}")
    
    if results['fit_pulses']:
        # Get Npulse statistics
        npulse_idx = get_npulse_index(model_name, results['max_peaks'], results['fit_pulses'])
        if npulse_idx is not None:
            npulse_samples = results['particles'][:, npulse_idx]
            mean_npulse = np.mean(npulse_samples)
            std_npulse = np.std(npulse_samples)
            mode_npulse = np.round(np.median(npulse_samples))
            print(f"Number of pulses: {mean_npulse:.2f} ± {std_npulse:.2f} (mode: {int(mode_npulse)})")
    
    print("="*50)