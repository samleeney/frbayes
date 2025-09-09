"""
, 
Test with simulated data: 2 pulses with fitted number of pulses.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import emg_model, get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling, save_chains_for_anesthetic
from frbayes_jax.analysis import analyze_results
from frbayes_jax.utils import get_default_prior_ranges


def main():
    """
    Test nested sampling with 2 pulses, fitting the number.
    """
    print("="*60)
    print("TEST: 2 PULSES WITH FITTED NUMBER")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 4  # Allow up to 4 peaks to test model selection
    fit_pulses = True  # Fit number of pulses
    
    # True parameters for 2 EMG pulses (but we'll search for up to 4)
    # For simulation, we need parameters for 2 pulses
    true_params_2peaks = jnp.array([
        0.8, 0.5,      # Amplitudes
        0.5, 0.3,      # Tau values
        1.0, 2.5,      # Arrival times (sorted)
        0.15, 0.12,    # Widths
        0.05           # Sigma (noise)
    ])
    
    print("\nTrue parameters (2 pulses):")
    print(f"  A1={true_params_2peaks[0]:.2f}, A2={true_params_2peaks[1]:.2f}")
    print(f"  tau1={true_params_2peaks[2]:.2f}, tau2={true_params_2peaks[3]:.2f}")
    print(f"  u1={true_params_2peaks[4]:.2f}, u2={true_params_2peaks[5]:.2f}")
    print(f"  w1={true_params_2peaks[6]:.2f}, w2={true_params_2peaks[7]:.2f}")
    print(f"  sigma={true_params_2peaks[8]:.3f}")
    print(f"  True Npulse=2")
    
    # Simulate data with 2 pulses
    print("\nSimulating data...")
    model_func_2peaks = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func_2peaks, true_params_2peaks, 2, False,  # Use 2 peaks for simulation
        t_min=0.0, t_max=4.0, num_points=500,
        add_noise=True, seed=42
    )
    
    # Convert to numpy for compatibility
    t_np = np.array(t)
    data_np = np.array(data)
    
    # Plot simulated data
    plt.figure(figsize=(10, 5))
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Simulated data')
    
    # Plot true model
    true_model = model_func_2peaks(t, true_params_2peaks, 2, False)
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, label='True model (2 pulses)')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Simulated Data: 2 Pulses (Fitted Number)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_2pulses_fitted_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to test_2pulses_fitted_data.png")
    
    # Set up priors (for up to 4 peaks)
    prior_ranges = get_default_prior_ranges(model_name)
    
    # Make priors a bit wider around true values
    prior_ranges["amplitude"]["min"] = 0.01
    prior_ranges["amplitude"]["max"] = 2.0
    prior_ranges["tau"]["min"] = 0.1
    prior_ranges["tau"]["max"] = 1.0
    prior_ranges["u"]["min"] = 0.0
    prior_ranges["u"]["max"] = 4.0
    prior_ranges["width"]["min"] = 0.01
    prior_ranges["width"]["max"] = 0.5
    prior_ranges["sigma"]["min"] = 0.001
    prior_ranges["sigma"]["max"] = 0.2
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    print(f"  Model: {model_name}")
    print(f"  Max peaks: {max_peaks} (searching for best number)")
    print(f"  Fit pulses: {fit_pulses}")
    
    results = run_nested_sampling(
        model_name=model_name,
        data=data_np,
        t=t_np,
        prior_ranges=prior_ranges,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=300,
        num_delete=20,
        num_inner_steps=5,
        max_iterations=500,
        log_tolerance=-1.0,
        seed=123,
        verbose=True
    )
    
    # Create output directory
    output_dir = "results_2pulses_fitted"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze results
    print("\nAnalyzing results...")
    chain_file = os.path.join(output_dir, "chains")
    analyze_results(
        results=results,
        model_name=model_name,
        t=t_np,
        data=data_np,
        output_dir=output_dir,
        chain_file=chain_file
    )
    
    # Extract best-fit parameters (using posterior mean)
    particles = results['particles']
    weights = np.exp(results['logL'] - np.max(results['logL']))
    weights = weights / np.sum(weights)
    
    best_fit = np.average(particles, axis=0, weights=weights)
    std_fit = np.sqrt(np.average((particles - best_fit)**2, axis=0, weights=weights))
    
    print("\nBest-fit parameters (posterior mean ± std):")
    param_names = get_param_names(model_name, max_peaks, fit_pulses)
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_fit[i]:.3f} ± {std_fit[i]:.3f}")
    
    # Get Npulse statistics
    npulse_idx = -1  # Last parameter
    npulse_samples = particles[:, npulse_idx]
    npulse_rounded = np.round(npulse_samples).astype(int)
    
    # Count occurrences
    unique, counts = np.unique(npulse_rounded, return_counts=True)
    weighted_counts = np.zeros_like(counts, dtype=float)
    for i, n in enumerate(unique):
        mask = npulse_rounded == n
        weighted_counts[i] = np.sum(weights[mask])
    
    print("\nNumber of pulses distribution:")
    for n, w in zip(unique, weighted_counts):
        print(f"  Npulse={n}: {w*100:.1f}%")
    
    most_probable_npulse = unique[np.argmax(weighted_counts)]
    print(f"\nMost probable number of pulses: {most_probable_npulse}")
    print(f"True number of pulses: 2")
    
    # Compare fitted parameters for first 2 pulses
    print("\nComparison with true parameters (first 2 pulses):")
    print(f"  A1: true={true_params_2peaks[0]:.3f}, fit={best_fit[0]:.3f} ± {std_fit[0]:.3f}")
    print(f"  A2: true={true_params_2peaks[1]:.3f}, fit={best_fit[1]:.3f} ± {std_fit[1]:.3f}")
    print(f"  tau1: true={true_params_2peaks[2]:.3f}, fit={best_fit[max_peaks]:.3f} ± {std_fit[max_peaks]:.3f}")
    print(f"  tau2: true={true_params_2peaks[3]:.3f}, fit={best_fit[max_peaks+1]:.3f} ± {std_fit[max_peaks+1]:.3f}")
    print(f"  u1: true={true_params_2peaks[4]:.3f}, fit={best_fit[2*max_peaks]:.3f} ± {std_fit[2*max_peaks]:.3f}")
    print(f"  u2: true={true_params_2peaks[5]:.3f}, fit={best_fit[2*max_peaks+1]:.3f} ± {std_fit[2*max_peaks+1]:.3f}")
    print(f"  w1: true={true_params_2peaks[6]:.3f}, fit={best_fit[3*max_peaks]:.3f} ± {std_fit[3*max_peaks]:.3f}")
    print(f"  w2: true={true_params_2peaks[7]:.3f}, fit={best_fit[3*max_peaks+1]:.3f} ± {std_fit[3*max_peaks+1]:.3f}")
    print(f"  sigma: true={true_params_2peaks[8]:.3f}, fit={best_fit[4*max_peaks]:.3f} ± {std_fit[4*max_peaks]:.3f}")
    
    # Plot best-fit model
    model_func = get_model_function(model_name)
    
    plt.figure(figsize=(12, 5))
    
    # Left panel: Data and models
    plt.subplot(1, 2, 1)
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Data')
    
    # True model
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, alpha=0.7, label='True model (2 pulses)')
    
    # Best-fit model
    best_fit_model = model_func(t, jnp.array(best_fit), max_peaks, fit_pulses)
    plt.plot(t_np, np.array(best_fit_model), 'b-', linewidth=2, alpha=0.7, 
             label=f'Best-fit model (Npulse≈{best_fit[-1]:.1f})')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right panel: Npulse distribution
    plt.subplot(1, 2, 2)
    plt.bar(unique, weighted_counts, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(2, color='red', linestyle='--', linewidth=2, label='True Npulse=2')
    plt.xlabel('Number of Pulses')
    plt.ylabel('Posterior Probability')
    plt.title('Pulse Number Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(1, max_peaks+1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_and_npulse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nModel comparison plot saved to {output_dir}/model_comparison_and_npulse.png")
    
    # Check if the model correctly identified 2 pulses
    if most_probable_npulse == 2:
        print("\n✓ SUCCESS: Model correctly identified 2 pulses!")
    else:
        print(f"\n⚠ WARNING: Model identified {most_probable_npulse} pulses instead of 2")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()