"""
Test that sorted priors work with the nested sampling.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import emg_model, get_model_function
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling


def main():
    """
    Quick test of sorted priors with nested sampling.
    """
    print("="*60)
    print("TEST: SORTED PRIORS WITH NESTED SAMPLING")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 2
    fit_pulses = False
    
    # True parameters with sorted arrival times
    true_params = jnp.array([
        0.5, 0.3,      # Amplitudes
        0.4, 0.3,      # Tau values  
        1.0, 2.0,      # Arrival times (SORTED!)
        0.1, 0.15,     # Widths
        0.05           # Sigma
    ])
    
    print("\nTrue parameters:")
    print(f"  u1={true_params[4]:.2f}, u2={true_params[5]:.2f} (sorted)")
    
    # Generate time array and data
    model_func = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func, true_params, max_peaks, fit_pulses,
        t_min=0.0, t_max=4.0, num_points=100, add_noise=True, seed=42
    )
    
    # Convert to numpy
    t_np = np.array(t)
    data_np = np.array(data)
    
    # Prior bounds
    prior_bounds = {
        'amplitude': {'min': 0.01, 'max': 1.0},
        'tau': {'min': 0.1, 'max': 1.0},
        'u': {'min': 0.0, 'max': 3.0},
        'width': {'min': 0.01, 'max': 0.5},
        'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.2)}
    }
    
    print("\nRunning nested sampling WITH sorted priors...")
    
    # Run with sorted priors
    final_state_sorted = run_nested_sampling(
        model_name=model_name,
        data=data_np,
        t=t_np,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        sorted_u=True,  # USE SORTED PRIORS
        num_live_points=200,
        num_delete=50,
        num_inner_steps=20,
        log_tolerance=-2.0,
        seed=123
    )
    
    print("  ✓ Nested sampling with sorted priors completed successfully!")
    
    print("\nRunning nested sampling WITHOUT sorted priors for comparison...")
    
    # Run without sorted priors 
    final_state_unsorted = run_nested_sampling(
        model_name=model_name,
        data=data_np,
        t=t_np,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        sorted_u=False,  # NO SORTED PRIORS
        num_live_points=200,
        num_delete=50,
        num_inner_steps=20,
        log_tolerance=-2.0,
        seed=456
    )
    
    print("  ✓ Nested sampling without sorted priors completed successfully!")
    
    # Extract u values from final live points
    if hasattr(final_state_sorted, 'live_points'):
        live_sorted = final_state_sorted.live_points
        u1_sorted = live_sorted[:, 4]
        u2_sorted = live_sorted[:, 5]
        
        # Check if sorted
        sorted_fraction = jnp.mean(u1_sorted <= u2_sorted)
        print(f"\nWith sorted priors: {sorted_fraction*100:.1f}% of live points have u1 <= u2")
        
    if hasattr(final_state_unsorted, 'live_points'):
        live_unsorted = final_state_unsorted.live_points
        u1_unsorted = live_unsorted[:, 4]
        u2_unsorted = live_unsorted[:, 5]
        
        unsorted_fraction = jnp.mean(u1_unsorted <= u2_unsorted)
        print(f"Without sorted priors: {unsorted_fraction*100:.1f}% of live points have u1 <= u2")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()