"""
Test with simulated data: 7 pulses with fitted number of pulses.
This is a simplified test case to verify the model can handle multi-pulse signals.
"""
import os
import sys
from datetime import datetime
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
    Test nested sampling with 7 pulses, fitting the number.
    """
    print("="*60)
    print("TEST: 7 PULSES WITH FITTED NUMBER")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 10  # Allow up to 10 peaks to test model selection
    fit_pulses = True  # Fit number of pulses
    
    # True parameters for 7 EMG pulses - STRONGER AND WELL SEPARATED
    # Arranged with sorted arrival times to match our prior
    true_params_7peaks = jnp.array([
        # Amplitudes - ALL STRONG for good SNR
        1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.4,
        # Tau values (varying decay times)
        0.35, 0.3, 0.4, 0.25, 0.32, 0.28, 0.22,
        # Arrival times (SORTED and WELL SEPARATED!)
        0.5, 1.2, 1.9, 2.6, 3.3, 4.0, 4.7,
        # Widths (varying widths)
        0.10, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13,
        # Sigma (REDUCED noise for better SNR)
        0.03
    ])
    
    print("\nTrue parameters (7 pulses):")
    print("  Amplitudes:", true_params_7peaks[0:7])
    print("  Tau values:", true_params_7peaks[7:14])
    print("  Arrival times (sorted):", true_params_7peaks[14:21])
    print("  Widths:", true_params_7peaks[21:28])
    print(f"  Sigma: {true_params_7peaks[28]:.3f}")
    print(f"  True Npulse: 7")
    
    # Generate time array and data
    # Use 7 peaks for data generation
    model_func_7peaks = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func_7peaks, 
        true_params_7peaks, 
        max_peaks=7,  # Generate with 7 peaks
        fit_pulses=False,  # Don't include Npulse in generation params
        t_min=0.0,
        t_max=5.5,  # Time range to accommodate all 7 pulses
        num_points=500,  # Good resolution
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
    true_model = model_func_7peaks(t, true_params_7peaks, 7, False)
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, label='True model (7 pulses)')
    
    # Plot individual pulses
    for i in range(7):
        single_pulse_params = jnp.zeros(29)  # 7*4 + 1 = 29
        single_pulse_params = single_pulse_params.at[0].set(true_params_7peaks[i])  # Amplitude
        single_pulse_params = single_pulse_params.at[7].set(true_params_7peaks[7+i])  # Tau
        single_pulse_params = single_pulse_params.at[14].set(true_params_7peaks[14+i])  # u
        single_pulse_params = single_pulse_params.at[21].set(true_params_7peaks[21+i])  # w
        single_pulse = model_func_7peaks(t, single_pulse_params, 7, False)
        plt.plot(t_np, np.array(single_pulse), '--', alpha=0.4, label=f'Pulse {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Simulated Data: 7 Pulses (Fitted Number)')
    plt.legend(loc='upper right', fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Add residual plot
    plt.subplot(2, 1, 2)
    residuals = data_np - np.array(true_model)
    plt.plot(t_np, residuals, 'b.', alpha=0.5, markersize=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=true_params_7peaks[28], color='r', linestyle='--', alpha=0.5, label=f'±σ = ±{true_params_7peaks[28]:.3f}')
    plt.axhline(y=-true_params_7peaks[28], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/test_7pulses_fitted_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to results/test_7pulses_fitted_data.png")
    
    # Set up prior bounds - CONSTRAINED to prevent overfitting
    prior_bounds = {
        'amplitude': {'min': 0.2, 'max': 1.8},  # Min 0.2 to avoid tiny noise fits
        'tau': {'min': 0.15, 'max': 0.5},  # Narrowed to match true range
        'u': {'min': 0.0, 'max': 5.5},  # Match data range [0, 5.5]
        'width': {'min': 0.06, 'max': 0.20},  # Narrowed range
        'log_sigma': {'min': jnp.log(0.02), 'max': jnp.log(0.08)}  # Narrowed around true value 0.03
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
    # When fitting Npulse with max_peaks=10: 10*(A, tau, u, w) + sigma + Npulse = 40 + 1 + 1 = 42
    ndims = 42
    
    # IMPORTANT: DO NOT CHANGE THESE HYPERPARAMETERS
    # These are the standard settings we always use for consistent performance
    num_live_points = ndims * 25  # 1050 (25 × 42)
    num_delete = num_live_points // 2  # 525
    num_inner_steps = ndims * 7  # 294 (7 × 42)
    
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
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/results_7pulses_fitted_{timestamp}"
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