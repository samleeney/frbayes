"""
Test with simulated data: periodic exponential pulses with fitted number of pulses.
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

from frbayes_jax.models import periodic_exponential_model, get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling


def main():
    """
    Test nested sampling with periodic exponential pulses, fitting the number.
    """
    print("="*60)
    print("TEST: PERIODIC EXPONENTIAL PULSES WITH FITTED NUMBER")
    print("="*60)
    
    # Settings
    model_name = "periodic_exponential"
    max_peaks = 4  # Allow up to 4 peaks to test model selection
    fit_pulses = True  # Fit number of pulses
    
    # True parameters for 3 periodic exponential pulses
    # Format: [A1, A2, A3, tau1, tau2, tau3, u0, period, sigma]
    true_npulse = 3
    true_params_3peaks = jnp.array([
        0.8, 0.6, 0.4,   # Amplitudes (decreasing)
        0.4, 0.3, 0.35,  # Tau values
        0.5,             # u0 (first pulse location)
        1.2,             # period (spacing between pulses)
        0.05             # Sigma (noise)
    ])
    
    print("\nTrue parameters (3 periodic pulses):")
    print(f"  A1={true_params_3peaks[0]:.2f}, A2={true_params_3peaks[1]:.2f}, A3={true_params_3peaks[2]:.2f}")
    print(f"  tau1={true_params_3peaks[3]:.2f}, tau2={true_params_3peaks[4]:.2f}, tau3={true_params_3peaks[5]:.2f}")
    print(f"  u0={true_params_3peaks[6]:.2f} (first pulse location)")
    print(f"  period={true_params_3peaks[7]:.2f}")
    print(f"  Pulse locations: u1={true_params_3peaks[6]:.2f}, u2={true_params_3peaks[6] + true_params_3peaks[7]:.2f}, u3={true_params_3peaks[6] + 2*true_params_3peaks[7]:.2f}")
    print(f"  sigma={true_params_3peaks[8]:.3f}")
    print(f"  True Npulse={true_npulse}")
    
    # Simulate data with 3 periodic pulses
    print("\nSimulating data...")
    model_func_3peaks = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func_3peaks, true_params_3peaks, 3, False,  # Use 3 peaks for simulation
        t_min=0.0, t_max=5.0, num_points=500,
        add_noise=True, seed=42
    )
    
    # Convert to numpy for compatibility
    t_np = np.array(t)
    data_np = np.array(data)
    
    # Plot simulated data
    plt.figure(figsize=(10, 5))
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Simulated data')
    
    # Plot true model
    true_model = model_func_3peaks(t, true_params_3peaks, 3, False)
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, label='True model (3 periodic pulses)')
    
    # Add vertical lines for pulse locations
    for i in range(3):
        pulse_loc = true_params_3peaks[6] + i * true_params_3peaks[7]
        plt.axvline(pulse_loc, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Simulated Data: Periodic Exponential Pulses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/test_periodic_exponential_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to results/test_periodic_exponential_data.png")
    
    # Set up prior bounds for periodic model
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 1},  # Wide amplitude range
        'tau': {'min': 0.1, 'max': 1.0},  # Tau range
        'u0': {'min': 0.0, 'max': 2.0},  # First pulse location
        'period': {'min': 0.5, 'max': 2.0},  # Period between pulses
        'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(2.0)}  # Log-uniform for sigma
    }
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    print(f"  Model: {model_name}")
    print(f"  Max peaks: {max_peaks} (searching for best number)")
    print(f"  Fit pulses: {fit_pulses}")
    
    # Calculate proper nested sampling parameters
    # When fitting Npulse: 4*(A, tau) + u0 + period + sigma + Npulse = 8 + 2 + 1 + 1 = 12
    ndims = 12
    num_live_points = ndims * 25  # 300
    num_delete = num_live_points // 2  # 150
    num_inner_steps = ndims * 5  # 60
    
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
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/results_periodic_exponential_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter names
    param_names = get_param_names(model_name, max_peaks, fit_pulses)
    
    # Save chains in anesthetic format
    print("\nSaving chains in anesthetic format...")
    # Save the chain data with dead-birth format
    chain_path = os.path.join(output_dir, 'chains')
    os.makedirs(os.path.dirname(chain_path) if os.path.dirname(chain_path) else '.', exist_ok=True)
    
    # Save particles, logL, and logL_birth
    chain_data = np.column_stack([
        final_state.particles,
        final_state.loglikelihood,
        final_state.loglikelihood_birth
    ])
    np.savetxt(f"{chain_path}_dead-birth.txt", chain_data)
    print(f"Chains saved to {chain_path}_dead-birth.txt")
    
    # Create NestedSamples object for analysis
    nested_samples = anesthetic.NestedSamples(
        data=final_state.particles,
        logL=final_state.loglikelihood,
        logL_birth=final_state.loglikelihood_birth,
        columns=param_names  # Add column names
    )
    
    # Get posterior statistics from anesthetic
    best_fit = nested_samples.mean().values
    std_fit = nested_samples.std().values
    
    print("\nBest-fit parameters (posterior mean ± std):")
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_fit[i]:.3f} ± {std_fit[i]:.3f}")
    
    # Extract the fitted number of pulses
    npulse_param_name = param_names[-1]  # Last parameter when fit_pulses=True
    fitted_npulse_mean = nested_samples[npulse_param_name].mean()
    fitted_npulse_std = nested_samples[npulse_param_name].std()
    print(f"\nFitted number of pulses: {fitted_npulse_mean:.2f} ± {fitted_npulse_std:.2f}")
    print(f"True number of pulses: {true_npulse}")
    
    # Calculate pulse locations from fitted parameters
    u0_idx = 2 * max_peaks  # Index of u0 in parameters
    period_idx = 2 * max_peaks + 1  # Index of period
    fitted_u0 = best_fit[u0_idx]
    fitted_period = best_fit[period_idx]
    
    print("\nFitted pulse locations:")
    for i in range(int(round(fitted_npulse_mean))):
        fitted_loc = fitted_u0 + i * fitted_period
        true_loc = true_params_3peaks[6] + i * true_params_3peaks[7] if i < true_npulse else None
        if true_loc is not None:
            print(f"  Pulse {i+1}: fitted={fitted_loc:.3f}, true={true_loc:.3f}")
        else:
            print(f"  Pulse {i+1}: fitted={fitted_loc:.3f}")
    
    # Model function for fitted parameters (use max_peaks for evaluation)
    model_func = get_model_function(model_name)
    
    # Plot best-fit model
    plt.figure(figsize=(10, 5))
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Data')
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, alpha=0.7, label=f'True model ({true_npulse} periodic pulses)')
    
    # Best-fit model with fitted number of pulses
    best_fit_model = model_func(t, jnp.array(best_fit), max_peaks, fit_pulses)
    plt.plot(t_np, np.array(best_fit_model), 'b-', linewidth=2, alpha=0.7, 
             label=f'Best-fit model (N≈{fitted_npulse_mean:.1f} periodic pulses)')
    
    # Add vertical lines for fitted pulse locations
    for i in range(int(round(fitted_npulse_mean))):
        pulse_loc = fitted_u0 + i * fitted_period
        plt.axvline(pulse_loc, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Model Comparison: Periodic Exponential Pulses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nModel comparison plot saved to {output_dir}/model_comparison.png")
    
    # Save parameter names for later use
    with open(os.path.join(output_dir, 'param_names.txt'), 'w') as f:
        for name in param_names:
            f.write(f"{name}\n")
    print(f"Parameter names saved to {output_dir}/param_names.txt")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'max_peaks': max_peaks,
        'fit_pulses': fit_pulses,
        'num_params': len(param_names),
        'num_samples': len(final_state.particles),
        'fitted_period': float(fitted_period),
        'fitted_u0': float(fitted_u0)
    }
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {output_dir}/metadata.json")
    
    # Check model selection performance
    print("\n" + "="*60)
    print("MODEL SELECTION RESULTS")
    print("="*60)
    
    # Calculate probability of different numbers of pulses
    npulse_samples = final_state.particles[:, -1]
    for n in range(1, max_peaks + 1):
        prob = np.mean((npulse_samples > n - 0.5) & (npulse_samples <= n + 0.5))
        indicator = " <-- TRUE" if n == true_npulse else ""
        print(f"  P(Npulse={n}) = {prob:.3f}{indicator}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()