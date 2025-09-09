"""
Verification test that both sampling cases work correctly.
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


def test_fixed_npulses():
    """Test with fixed number of pulses."""
    print("\n" + "="*60)
    print("TEST 1: FIXED NUMBER OF PULSES (2)")
    print("="*60)
    
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
    
    # Simulate data
    model_func = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func, true_params, max_peaks, fit_pulses,
        t_min=0.0, t_max=4.0, num_points=100,
        add_noise=True, seed=42
    )
    
    # Set up priors
    prior_ranges = get_default_prior_ranges(model_name)
    prior_ranges["amplitude"]["max"] = 2.0
    prior_ranges["u"]["max"] = 4.0
    prior_ranges["sigma"]["max"] = 0.2
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    results = run_nested_sampling(
        model_name=model_name,
        data=np.array(data),
        t=np.array(t),
        prior_ranges=prior_ranges,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=200,
        num_delete=10,
        num_inner_steps=5,
        max_iterations=100,
        log_tolerance=-0.5,
        seed=123,
        verbose=False
    )
    
    print(f"✓ Sampling completed")
    print(f"  Log evidence: {results['logZ']:.2f}")
    print(f"  Number of samples: {results['nsamples']}")
    
    # Get best fit
    particles = results['particles']
    weights = np.exp(results['logL'] - np.max(results['logL']))
    weights = weights / np.sum(weights)
    best_fit = np.average(particles, axis=0, weights=weights)
    
    print(f"\nParameter recovery:")
    print(f"  A1: true={true_params[0]:.2f}, fit={best_fit[0]:.2f}")
    print(f"  A2: true={true_params[1]:.2f}, fit={best_fit[1]:.2f}")
    print(f"  sigma: true={true_params[8]:.3f}, fit={best_fit[8]:.3f}")
    
    return True


def test_fitted_npulses():
    """Test with fitted number of pulses."""
    print("\n" + "="*60)
    print("TEST 2: FITTED NUMBER OF PULSES")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 4  # Allow up to 4
    fit_pulses = True
    
    # True parameters for 2 pulses
    true_params_2peaks = jnp.array([
        0.8, 0.5,      # Amplitudes
        0.5, 0.3,      # Tau values
        1.0, 2.5,      # Arrival times
        0.15, 0.12,    # Widths
        0.05           # Sigma
    ])
    
    print(f"True number of pulses: 2")
    
    # Simulate data with 2 pulses
    model_func_2peaks = get_model_function(model_name)
    t, data = simulate_frb_data(
        model_func_2peaks, true_params_2peaks, 2, False,
        t_min=0.0, t_max=4.0, num_points=100,
        add_noise=True, seed=42
    )
    
    # Set up priors
    prior_ranges = get_default_prior_ranges(model_name)
    prior_ranges["amplitude"]["max"] = 2.0
    prior_ranges["u"]["max"] = 4.0
    prior_ranges["sigma"]["max"] = 0.2
    
    # Run nested sampling
    print("\nRunning nested sampling (searching for best Npulse)...")
    results = run_nested_sampling(
        model_name=model_name,
        data=np.array(data),
        t=np.array(t),
        prior_ranges=prior_ranges,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=200,
        num_delete=10,
        num_inner_steps=5,
        max_iterations=100,
        log_tolerance=-0.5,
        seed=123,
        verbose=False
    )
    
    print(f"✓ Sampling completed")
    print(f"  Log evidence: {results['logZ']:.2f}")
    print(f"  Number of samples: {results['nsamples']}")
    
    # Get Npulse statistics
    npulse_idx = -1  # Last parameter
    npulse_samples = results['particles'][:, npulse_idx]
    npulse_rounded = np.round(npulse_samples).astype(int)
    
    # Count occurrences
    unique, counts = np.unique(npulse_rounded, return_counts=True)
    
    # Get weights
    weights = np.exp(results['logL'] - np.max(results['logL']))
    weights = weights / np.sum(weights)
    
    weighted_counts = np.zeros_like(counts, dtype=float)
    for i, n in enumerate(unique):
        mask = npulse_rounded == n
        weighted_counts[i] = np.sum(weights[mask])
    
    print(f"\nNumber of pulses distribution:")
    for n, w in zip(unique, weighted_counts):
        symbol = "★" if n == 2 else " "
        print(f"  {symbol} Npulse={n}: {w*100:.1f}%")
    
    most_probable_npulse = unique[np.argmax(weighted_counts)]
    
    if most_probable_npulse == 2:
        print(f"\n✓ SUCCESS: Model correctly identified 2 pulses!")
        return True
    else:
        print(f"\n⚠ Model identified {most_probable_npulse} pulses instead of 2")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print(" "*20 + "FRBAYES JAX VERIFICATION TESTS")
    print("="*70)
    
    tests_passed = []
    
    # Test 1: Fixed number of pulses
    try:
        result = test_fixed_npulses()
        tests_passed.append(("Fixed Npulse", result))
    except Exception as e:
        print(f"\n✗ Test 1 failed with error: {e}")
        tests_passed.append(("Fixed Npulse", False))
    
    # Test 2: Fitted number of pulses
    try:
        result = test_fitted_npulses()
        tests_passed.append(("Fitted Npulse", result))
    except Exception as e:
        print(f"\n✗ Test 2 failed with error: {e}")
        tests_passed.append(("Fitted Npulse", False))
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in tests_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(p for _, p in tests_passed)
    
    if all_passed:
        print("\n" + "="*70)
        print(" "*15 + "ALL TESTS PASSED SUCCESSFULLY!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(" "*20 + "SOME TESTS FAILED")
        print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)