"""
Test without JIT to debug the hanging issue.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Disable JIT for debugging
jax.config.update("jax_disable_jit", True)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import emg_model, get_model_function
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling
from frbayes_jax.utils import get_default_prior_ranges


def main():
    """
    Test without JIT.
    """
    print("="*60)
    print("TEST WITHOUT JIT")
    print("="*60)
    print("JAX JIT disabled for debugging")
    
    # Settings
    model_name = "emg"
    max_peaks = 2
    fit_pulses = False
    
    # True parameters
    true_params = jnp.array([
        0.8, 0.5,      # Amplitudes
        0.5, 0.3,      # Tau values
        1.0, 2.5,      # Arrival times
        0.15, 0.12,    # Widths
        0.05           # Sigma
    ])
    
    print("\nGenerating simulated data...")
    model_func = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func, true_params, max_peaks, fit_pulses,
        t_min=0.0, t_max=4.0, num_points=50,  # Very few points
        add_noise=True, seed=42
    )
    
    t_np = np.array(t)
    data_np = np.array(data)
    print(f"Data shape: {data_np.shape}")
    
    # Set up priors
    prior_ranges = get_default_prior_ranges(model_name)
    prior_ranges["amplitude"]["max"] = 2.0
    prior_ranges["u"]["max"] = 4.0
    prior_ranges["sigma"]["max"] = 0.2
    
    # Run nested sampling with minimal settings
    print("\nRunning nested sampling (no JIT)...")
    results = run_nested_sampling(
        model_name=model_name,
        data=data_np,
        t=t_np,
        prior_ranges=prior_ranges,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=50,  # Very few
        num_delete=5,
        num_inner_steps=2,  # Very few
        max_iterations=10,  # Just a few iterations
        log_tolerance=0.0,  # Don't converge
        seed=123,
        verbose=True
    )
    
    print("\nResults:")
    print(f"  Log evidence: {results['logZ']:.4f}")
    print(f"  Number of samples: {results['nsamples']}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()