"""
Test with simulated data: 2 pulses with fixed number of pulses.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import anesthetic
from anesthetic import make_2d_axes

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import emg_model, get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling
from frbayes_jax.analysis import analyze_results


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
    
    # Set up prior bounds (using the user's requested wide priors)
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 1},  # Very wide amplitude range
        'tau': {'min': 0.1, 'max': 1.0},  # Set to [0.1, 1.0] to avoid numerical issues
        'u': {'min': 0.0, 'max': 4.0},  # Match data range [0, 4]
        'width': {'min': 0.01, 'max': 0.3},  # Reduced max to avoid exp overflow with small tau
        'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(2.0)}  # Log-uniform for sigma
    }
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    print(f"  Model: {model_name}")
    print(f"  Max peaks: {max_peaks}")
    print(f"  Fit pulses: {fit_pulses}")
    
    # Calculate proper nested sampling parameters
    ndims = 9  # 2*(A, tau, u, w) + sigma = 8 + 1 = 9
    num_live_points = ndims * 25  # 225
    num_delete = num_live_points // 2  # 112
    num_inner_steps = ndims * 5  # 45
    
    # Run nested sampling and get final_state directly
    final_state = run_nested_sampling(
        model_name=model_name,
        data=data_np,
        t=t_np,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=num_live_points,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
        log_tolerance=-3.0,
        seed=123
    )
    
    
    print("\nNested sampling completed.")
    
    # Create output directory
    output_dir = "results_2pulses_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter names
    param_names = get_param_names(model_name, max_peaks, fit_pulses)
    
    # Create NestedSamples object
    print("\nCreating NestedSamples object...")
    nested_samples = anesthetic.NestedSamples(
        data=final_state.particles,
        logL=final_state.loglikelihood,
        logL_birth=final_state.loglikelihood_birth,  # Already fixed in sampling.py
    )
    
    # Get posterior statistics from anesthetic
    best_fit = nested_samples.mean().values
    std_fit = nested_samples.std().values
    
    print("\nBest-fit parameters (posterior mean ± std):")
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_fit[i]:.3f} ± {std_fit[i]:.3f}")
    
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
    
    # Create corner plot
    print("\nCreating corner plot...")
    try:
        # Plot first 5 parameters
        indices = np.arange(min(5, len(param_names)))
        fig, axes = nested_samples.plot_2d(indices)
        fig.savefig(os.path.join(output_dir, 'corner_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Corner plot saved to {output_dir}/corner_plot.png")
    except Exception as e:
        print(f"Warning: Could not create corner plot: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()