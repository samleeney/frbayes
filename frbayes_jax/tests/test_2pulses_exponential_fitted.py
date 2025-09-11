"""
Test with simulated data: 2 pulses with fitted number of pulses using exponential model.
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

from frbayes_jax.models import exponential_model, get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling


def main():
    """
    Test nested sampling with 2 pulses, fitting the number, using exponential model.
    """
    print("="*60)
    print("TEST: 2 PULSES WITH FITTED NUMBER (EXPONENTIAL MODEL)")
    print("="*60)
    
    # Settings
    model_name = "exponential"
    max_peaks = 4  # Allow up to 4 peaks to test model selection
    fit_pulses = True  # Fit number of pulses
    
    # True parameters for 2 exponential pulses (but we'll search for up to 4)
    # For simulation, we need parameters for 2 pulses
    true_params_2peaks = jnp.array([
        0.8, 0.5,      # Amplitudes
        0.5, 0.3,      # Tau values
        1.0, 2.5,      # Arrival times (sorted)
        0.05           # Sigma (noise)
    ])
    
    print("\nTrue parameters (2 pulses):")
    print(f"  A1={true_params_2peaks[0]:.2f}, A2={true_params_2peaks[1]:.2f}")
    print(f"  tau1={true_params_2peaks[2]:.2f}, tau2={true_params_2peaks[3]:.2f}")
    print(f"  u1={true_params_2peaks[4]:.2f}, u2={true_params_2peaks[5]:.2f}")
    print(f"  sigma={true_params_2peaks[6]:.3f}")
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
    plt.title('Simulated Data: 2 Pulses (Fitted Number, Exponential Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_2pulses_exponential_fitted_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Data plot saved to test_2pulses_exponential_fitted_data.png")
    
    # Set up prior bounds (using the user's requested wide priors)
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 1},  # Very wide amplitude range
        'tau': {'min': 0.1, 'max': 1.0},  # Set to [0.1, 1.0] to avoid numerical issues
        'u': {'min': 0.0, 'max': 4.0},  # Match data range [0, 4]
        'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(2.0)}  # Log-uniform for sigma
    }
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    print(f"  Model: {model_name}")
    print(f"  Max peaks: {max_peaks} (searching for best number)")
    print(f"  Fit pulses: {fit_pulses}")
    
    # Calculate proper nested sampling parameters
    # When fitting Npulse: 4*(A, tau, u) + sigma + Npulse = 12 + 1 + 1 = 14
    ndims = 14
    num_live_points = ndims * 25  # 350
    num_delete = num_live_points // 2  # 175
    num_inner_steps = ndims * 5  # 70
    
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
    output_dir = "results_2pulses_exponential_fitted"
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
    print(f"True number of pulses: 2")
    
    # Model function for fitted parameters (use max_peaks for evaluation)
    model_func = get_model_function(model_name)
    
    # Plot best-fit model
    plt.figure(figsize=(10, 5))
    plt.plot(t_np, data_np, 'k.', alpha=0.5, markersize=2, label='Data')
    plt.plot(t_np, np.array(true_model), 'r-', linewidth=2, alpha=0.7, label='True model (2 pulses)')
    
    # Best-fit model with fitted number of pulses
    best_fit_model = model_func(t, jnp.array(best_fit), max_peaks, fit_pulses)
    plt.plot(t_np, np.array(best_fit_model), 'b-', linewidth=2, alpha=0.7, 
             label=f'Best-fit model (N≈{fitted_npulse_mean:.1f})')
    
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Model Comparison: 2 Pulses (Fitted Number, Exponential Model)')
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
        'num_samples': len(final_state.particles)
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
        indicator = " <-- TRUE" if n == 2 else ""
        print(f"  P(Npulse={n}) = {prob:.3f}{indicator}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()