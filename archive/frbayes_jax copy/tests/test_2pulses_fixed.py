"""
Test with simulated data: 2 pulses with fixed number of pulses.
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
    Test nested sampling with 2 pulses, fixed number.
    """
    print("="*60)
    print("TEST: 2 PULSES WITH FIXED NUMBER")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 2
    fit_pulses = False  # Fixed number of pulses
    
    # True parameters for 2 EMG pulses
    # [A1, A2, tau1, tau2, u1, u2, w1, w2, sigma]
    true_params = jnp.array([
        0.8, 0.5,      # Amplitudes
        0.5, 0.3,      # Tau values
        1.0, 2.5,      # Arrival times (sorted)
        0.15, 0.12,    # Widths
        0.05           # Sigma (noise)
    ])
    
    print("\nTrue parameters:")
    print(f"  A1={true_params[0]:.2f}, A2={true_params[1]:.2f}")
    print(f"  tau1={true_params[2]:.2f}, tau2={true_params[3]:.2f}")
    print(f"  u1={true_params[4]:.2f}, u2={true_params[5]:.2f}")
    print(f"  w1={true_params[6]:.2f}, w2={true_params[7]:.2f}")
    print(f"  sigma={true_params[8]:.3f}")
    
    # Simulate data
    print("\nSimulating data...")
    model_func = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func, true_params, max_peaks, fit_pulses,
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
    true_model = model_func(t, true_params, max_peaks, fit_pulses)
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, label='True model')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Simulated Data: 2 Pulses (Fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_2pulses_fixed_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to test_2pulses_fixed_data.png")
    
    # Set up priors
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
    print(f"  Max peaks: {max_peaks}")
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
    output_dir = "results_2pulses_fixed"
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
    
    # Compare with true parameters
    print("\nComparison with true parameters:")
    print(f"  A1: true={true_params[0]:.3f}, fit={best_fit[0]:.3f} ± {std_fit[0]:.3f}")
    print(f"  A2: true={true_params[1]:.3f}, fit={best_fit[1]:.3f} ± {std_fit[1]:.3f}")
    print(f"  tau1: true={true_params[2]:.3f}, fit={best_fit[2]:.3f} ± {std_fit[2]:.3f}")
    print(f"  tau2: true={true_params[3]:.3f}, fit={best_fit[3]:.3f} ± {std_fit[3]:.3f}")
    print(f"  u1: true={true_params[4]:.3f}, fit={best_fit[4]:.3f} ± {std_fit[4]:.3f}")
    print(f"  u2: true={true_params[5]:.3f}, fit={best_fit[5]:.3f} ± {std_fit[5]:.3f}")
    print(f"  w1: true={true_params[6]:.3f}, fit={best_fit[6]:.3f} ± {std_fit[6]:.3f}")
    print(f"  w2: true={true_params[7]:.3f}, fit={best_fit[7]:.3f} ± {std_fit[7]:.3f}")
    print(f"  sigma: true={true_params[8]:.3f}, fit={best_fit[8]:.3f} ± {std_fit[8]:.3f}")
    
    # Plot best-fit model
    plt.figure(figsize=(10, 5))
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Data')
    
    # True model
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, alpha=0.7, label='True model')
    
    # Best-fit model
    best_fit_model = model_func(t, jnp.array(best_fit), max_peaks, fit_pulses)
    plt.plot(t_np, np.array(best_fit_model), 'b-', linewidth=2, alpha=0.7, label='Best-fit model')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Model Comparison: 2 Pulses (Fixed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nModel comparison plot saved to {output_dir}/model_comparison.png")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()