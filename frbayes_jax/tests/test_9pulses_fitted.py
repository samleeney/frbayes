"""
Test with simulated data: 9 pulses with fitted number of pulses.
This is a challenging test case to verify the model can handle complex multi-pulse signals.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import anesthetic

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import emg_model, get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling


def main():
    """
    Test nested sampling with 9 pulses, fitting the number.
    """
    print("="*60)
    print("TEST: 9 PULSES WITH FITTED NUMBER")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 14  # Allow up to 14 peaks to test model selection
    fit_pulses = True  # Fit number of pulses
    
    # True parameters for 9 EMG pulses
    # Arranged with sorted arrival times to match our prior
    true_params_9peaks = jnp.array([
        # Amplitudes (decreasing trend with some variation)
        1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3,
        # Tau values (varying decay times)
        0.35, 0.3, 0.4, 0.25, 0.32, 0.28, 0.22, 0.3, 0.25,
        # Arrival times (SORTED!)
        0.3, 0.8, 1.3, 1.9, 2.5, 3.1, 3.7, 4.3, 4.8,
        # Widths (varying widths)
        0.10, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.10, 0.08,
        # Sigma (noise)
        0.05
    ])
    
    print("\nTrue parameters (9 pulses):")
    print("  Amplitudes:", true_params_9peaks[0:9])
    print("  Tau values:", true_params_9peaks[9:18])
    print("  Arrival times (sorted):", true_params_9peaks[18:27])
    print("  Widths:", true_params_9peaks[27:36])
    print(f"  Sigma: {true_params_9peaks[36]:.3f}")
    print(f"  True Npulse: 9")
    
    # Generate time array and data
    # Use 9 peaks for data generation
    model_func_9peaks = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func_9peaks, 
        true_params_9peaks, 
        max_peaks=9,  # Generate with 9 peaks
        fit_pulses=False,  # Don't include Npulse in generation params
        t_min=0.0,
        t_max=5.5,  # Slightly longer to accommodate all 9 pulses
        num_points=250,  # More points for better resolution
        add_noise=True,
        seed=42
    )
    
    print("\nSimulated data generated")
    print(f"  Time range: [{np.min(t):.1f}, {np.max(t):.1f}]")
    print(f"  Number of points: {len(t)}")
    print(f"  Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Convert to numpy for compatibility
    t_np = np.array(t)
    data_np = np.array(data)
    
    # Plot simulated data
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Simulated data')
    
    # Plot true model
    true_model = model_func_9peaks(t, true_params_9peaks, 9, False)
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, label='True model (9 pulses)')
    
    # Plot individual pulses
    for i in range(9):
        single_pulse_params = jnp.zeros(37)  # 9*4 + 1 = 37
        single_pulse_params = single_pulse_params.at[0].set(true_params_9peaks[i])  # Amplitude
        single_pulse_params = single_pulse_params.at[9].set(true_params_9peaks[9+i])  # Tau
        single_pulse_params = single_pulse_params.at[18].set(true_params_9peaks[18+i])  # u
        single_pulse_params = single_pulse_params.at[27].set(true_params_9peaks[27+i])  # w
        single_pulse = model_func_9peaks(t, single_pulse_params, 9, False)
        plt.plot(t_np, np.array(single_pulse), '--', alpha=0.4, label=f'Pulse {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Simulated Data: 9 Pulses (Fitted Number)')
    plt.legend(loc='upper right', fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Add residual plot
    plt.subplot(2, 1, 2)
    residuals = data_np - np.array(true_model)
    plt.plot(t_np, residuals, 'b.', alpha=0.5, markersize=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=true_params_9peaks[36], color='r', linestyle='--', alpha=0.5, label=f'±σ = ±{true_params_9peaks[36]:.3f}')
    plt.axhline(y=-true_params_9peaks[36], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_9pulses_fitted_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to test_9pulses_fitted_data.png")
    
    # Set up prior bounds (wider to accommodate 9 pulses)
    prior_bounds = {
        'amplitude': {'min': 0.1, 'max': 1.5},  # Wide amplitude range
        'tau': {'min': 0.1, 'max': 0.8},  # Reasonable decay times
        'u': {'min': 0.0, 'max': 5.5},  # Match data range [0, 5.5]
        'width': {'min': 0.05, 'max': 0.25},  # Reasonable widths
        'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.2)}  # Log-uniform for sigma
    }
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    print(f"  Model: {model_name}")
    print(f"  Max peaks: {max_peaks} (searching for best number)")
    print(f"  Fit pulses: {fit_pulses}")
    print("  Prior bounds:")
    for key, bounds in prior_bounds.items():
        if key != 'log_sigma':
            print(f"    {key}: [{bounds['min']:.3f}, {bounds['max']:.3f}]")
        else:
            print(f"    sigma: [{np.exp(bounds['min']):.3f}, {np.exp(bounds['max']):.3f}]")
    
    # Calculate proper nested sampling parameters
    # When fitting Npulse with max_peaks=14: 14*(A, tau, u, w) + sigma + Npulse = 56 + 1 + 1 = 58
    ndims = 58
    
    # IMPORTANT: DO NOT CHANGE THESE HYPERPARAMETERS
    # These are the standard settings we always use for consistent performance
    num_live_points = ndims * 25  # 1450 (25 × 58)
    num_delete = num_live_points // 2  # ~181
    num_inner_steps = ndims * 7  # 290 (5 × 58)
    
    print(f"\nSampling parameters:")
    print(f"  Dimensions: {ndims}")
    print(f"  Live points: {num_live_points}")
    print(f"  Delete fraction: {num_delete}")
    print(f"  Inner steps: {num_inner_steps}")
    
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
        log_tolerance=-3.0,  # Standard tolerance (DO NOT CHANGE)
        seed=123
    )
    
    print("\nNested sampling completed.")
    
    # Create output directory
    output_dir = "results_9pulses_fitted"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter names
    param_names = get_param_names(model_name, max_peaks, fit_pulses)
    
    # Save chains in anesthetic format
    print("\nSaving chains in anesthetic format...")
    chain_path = os.path.join(output_dir, 'chains')
    
    # Convert to anesthetic samples
    import anesthetic
    samples = anesthetic.NestedSamples(
        data=final_state.particles,
        logL=final_state.loglikelihood,
        logL_birth=final_state.loglikelihood_birth,
        columns=param_names
    )
    
    # Save the samples
    samples.to_csv(chain_path + '_dead-birth.txt', sep=' ', index=False)
    print(f"Chains saved to {chain_path}_dead-birth.txt")
    
    # Analyze results
    print("\nBest-fit parameters (posterior mean ± std):")
    for i, name in enumerate(param_names):
        mean_val = samples[name].mean()
        std_val = samples[name].std()
        print(f"  {name}: {mean_val:.3f} ± {std_val:.3f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return final_state, samples


if __name__ == "__main__":
    final_state, samples = main()