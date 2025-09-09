"""
Simple test to verify the JAX implementation works.
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
from frbayes_jax.utils import get_default_prior_ranges


def main():
    """
    Simple test with minimal iterations.
    """
    print("="*60)
    print("SIMPLE TEST: VERIFY JAX IMPLEMENTATION")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 2
    fit_pulses = False
    
    # True parameters for 2 EMG pulses
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
        t_min=0.0, t_max=4.0, num_points=100,  # Fewer points for speed
        add_noise=True, seed=42
    )
    
    t_np = np.array(t)
    data_np = np.array(data)
    print(f"Data shape: {data_np.shape}")
    print(f"Data range: [{data_np.min():.3f}, {data_np.max():.3f}]")
    
    # Set up priors
    prior_ranges = get_default_prior_ranges(model_name)
    prior_ranges["amplitude"]["max"] = 2.0
    prior_ranges["u"]["max"] = 4.0
    prior_ranges["width"]["max"] = 0.5
    prior_ranges["sigma"]["max"] = 0.2
    
    # Run nested sampling with minimal settings
    print("\nRunning nested sampling (minimal test)...")
    results = run_nested_sampling(
        model_name=model_name,
        data=data_np,
        t=t_np,
        prior_ranges=prior_ranges,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=100,  # Very few for speed
        num_delete=10,
        num_inner_steps=5,
        max_iterations=50,  # Very few iterations
        log_tolerance=-1.0,  # Less strict
        seed=123,
        verbose=True
    )
    
    print("\nResults:")
    print(f"  Log evidence: {results['logZ']:.4f}")
    print(f"  Number of samples: {results['nsamples']}")
    print(f"  Particles shape: {results['particles'].shape}")
    
    # Get best fit
    particles = results['particles']
    weights = np.exp(results['logL'] - np.max(results['logL']))
    weights = weights / np.sum(weights)
    best_fit = np.average(particles, axis=0, weights=weights)
    
    print("\nBest-fit parameters (weighted mean):")
    print(f"  A1={best_fit[0]:.3f} (true={true_params[0]:.3f})")
    print(f"  A2={best_fit[1]:.3f} (true={true_params[1]:.3f})")
    print(f"  sigma={best_fit[8]:.3f} (true={true_params[8]:.3f})")
    
    print("\n" + "="*60)
    print("SIMPLE TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()