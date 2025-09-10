"""
Test with simulated data: 6 pulses with fitted number of pulses.
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
    Test nested sampling with 6 pulses, fitting the number.
    """
    print("="*60)
    print("TEST: 6 PULSES WITH FITTED NUMBER")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 8  # Allow up to 8 peaks to test model selection
    fit_pulses = True  # Fit number of pulses
    
    # True parameters for 6 EMG pulses
    # Arranged with sorted arrival times to match our prior
    true_params_6peaks = jnp.array([
        # Amplitudes (decreasing trend)
        1.0, 0.8, 0.6, 0.5, 0.4, 0.3,
        # Tau values (varying decay times)
        0.4, 0.3, 0.35, 0.25, 0.3, 0.2,
        # Arrival times (SORTED!)
        0.5, 1.2, 2.0, 2.8, 3.5, 4.2,
        # Widths (varying widths)
        0.12, 0.10, 0.15, 0.08, 0.11, 0.09,
        # Sigma (noise)
        0.05
    ])
    
    print("\nTrue parameters (6 pulses):")
    print("  Amplitudes:", true_params_6peaks[0:6])
    print("  Tau values:", true_params_6peaks[6:12])
    print("  Arrival times (sorted):", true_params_6peaks[12:18])
    print("  Widths:", true_params_6peaks[18:24])
    print(f"  Sigma: {true_params_6peaks[24]:.3f}")
    print(f"  True Npulse: 6")
    
    # Generate time array and data
    # Use 6 peaks for data generation
    model_func_6peaks = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func_6peaks, 
        true_params_6peaks, 
        max_peaks=6,  # Generate with 6 peaks
        fit_pulses=False,  # Don't include Npulse in generation params
        t_min=0.0,
        t_max=5.0,
        num_points=200,
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
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Simulated data')
    
    # Plot true model
    true_model = model_func_6peaks(t, true_params_6peaks, 6, False)
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, label='True model (6 pulses)')
    
    # Plot individual pulses
    for i in range(6):
        single_pulse_params = jnp.zeros(25)
        single_pulse_params = single_pulse_params.at[0].set(true_params_6peaks[i])  # Amplitude
        single_pulse_params = single_pulse_params.at[6].set(true_params_6peaks[6+i])  # Tau
        single_pulse_params = single_pulse_params.at[12].set(true_params_6peaks[12+i])  # u
        single_pulse_params = single_pulse_params.at[18].set(true_params_6peaks[18+i])  # w
        single_pulse = model_func_6peaks(t, single_pulse_params, 6, False)
        plt.plot(t_np, np.array(single_pulse), '--', alpha=0.5, label=f'Pulse {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Simulated Data: 6 Pulses (Fitted Number)')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Add residual plot
    plt.subplot(2, 1, 2)
    residuals = data_np - np.array(true_model)
    plt.plot(t_np, residuals, 'b.', alpha=0.5, markersize=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=true_params_6peaks[24], color='r', linestyle='--', alpha=0.5, label=f'±σ = ±{true_params_6peaks[24]:.3f}')
    plt.axhline(y=-true_params_6peaks[24], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_6pulses_fitted_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to test_6pulses_fitted_data.png")
    
    # Set up prior bounds (wider to accommodate 6 pulses)
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 1.5},  # Wide amplitude range
        'tau': {'min': 0.1, 'max': 0.8},  # Reasonable decay times
        'u': {'min': 0.0, 'max': 5.0},  # Match data range [0, 5]
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
    # When fitting Npulse with max_peaks=8: 8*(A, tau, u, w) + sigma + Npulse = 32 + 1 + 1 = 34
    ndims = 34
    
    # IMPORTANT: DO NOT CHANGE THESE HYPERPARAMETERS
    # These are the standard settings we always use for consistent performance
    num_live_points = ndims * 25  # 850 (25 × 34)
    num_delete = num_live_points // 8  # 425
    num_inner_steps = ndims * 5  # 170 (5 × 34)
    
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
    output_dir = "results_6pulses_fitted"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter names
    param_names = get_param_names(model_name, max_peaks, fit_pulses)
    
    # Save chains in anesthetic format
    print("\nSaving chains in anesthetic format...")
    chain_path = os.path.join(output_dir, 'chains')
    
    # Convert to anesthetic samples
    import anesthetic
    samples = anesthetic.NestedSamples(
        data=final_state.dead_points.params,
        logL=final_state.dead_points.logL,
        logL_birth=final_state.dead_points.logL_birth,
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
    
    # Check which Npulse value was preferred
    if fit_pulses:
        npulse_samples = samples['Npulse'].values
        npulse_mode = int(np.round(np.median(npulse_samples)))
        npulse_mean = np.mean(npulse_samples)
        npulse_std = np.std(npulse_samples)
        
        print(f"\nNumber of pulses analysis:")
        print(f"  True Npulse: 6")
        print(f"  Fitted Npulse (median): {npulse_mode}")
        print(f"  Fitted Npulse (mean ± std): {npulse_mean:.2f} ± {npulse_std:.2f}")
        
        # Histogram of Npulse
        plt.figure(figsize=(8, 5))
        plt.hist(npulse_samples, bins=np.arange(0.5, max_peaks+1.5, 1), 
                density=True, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=6, color='red', linestyle='--', linewidth=2, label='True value (6)')
        plt.axvline(x=npulse_mode, color='green', linestyle='-', linewidth=2, label=f'Fitted mode ({npulse_mode})')
        plt.xlabel('Number of pulses')
        plt.ylabel('Posterior probability')
        plt.title('Posterior Distribution of Number of Pulses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'npulse_posterior.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Npulse posterior plot saved to {output_dir}/npulse_posterior.png")
    
    # Plot best-fit model
    print("\nGenerating model comparison plot...")
    
    # Get best-fit parameters (posterior mean)
    best_fit_params = jnp.array([samples[name].mean() for name in param_names[:-1]])  # Exclude Npulse for model
    
    # Generate best-fit model
    model_func = get_model_function(model_name)
    best_fit_model = model_func(t, best_fit_params, max_peaks, False)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Data')
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, alpha=0.7, label='True model (6 pulses)')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('True Model vs Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Data')
    plt.plot(t_np, np.array(best_fit_model), 'b-', linewidth=2, alpha=0.7, label=f'Best fit ({npulse_mode} pulses)')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Best Fit Model vs Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    residuals_fit = data_np - np.array(best_fit_model)
    plt.plot(t_np, residuals_fit, 'g.', alpha=0.5, markersize=2, label='Fit residuals')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    fitted_sigma = samples['$\\sigma$'].mean()
    plt.axhline(y=fitted_sigma, color='b', linestyle='--', alpha=0.5, label=f'±σ_fit = ±{fitted_sigma:.3f}')
    plt.axhline(y=-fitted_sigma, color='b', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Residuals of Best Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {output_dir}/model_comparison.png")
    
    # Save parameter names
    with open(os.path.join(output_dir, 'param_names.txt'), 'w') as f:
        for name in param_names:
            f.write(f"{name}\n")
    print(f"Parameter names saved to {output_dir}/param_names.txt")
    
    # Save metadata
    import json
    metadata = {
        'model_name': model_name,
        'max_peaks': max_peaks,
        'true_npulse': 6,
        'fitted_npulse_mode': int(npulse_mode) if fit_pulses else None,
        'fit_pulses': fit_pulses,
        'num_live_points': num_live_points,
        'num_delete': num_delete,
        'num_inner_steps': num_inner_steps,
        'log_tolerance': -3.0,
        'seed': 123,
        'prior_bounds': prior_bounds,
        'time_range': [float(np.min(t)), float(np.max(t))],
        'num_data_points': len(t)
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {output_dir}/metadata.json")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    
    # Summary
    if fit_pulses:
        if npulse_mode == 6:
            print("✓ SUCCESS: Correctly identified 6 pulses!")
        else:
            print(f"⚠ Model selected {npulse_mode} pulses instead of true 6")
    
    return final_state, samples


if __name__ == "__main__":
    final_state, samples = main()