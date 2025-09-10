"""
Test sorted priors for arrival times.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.priors import FRBPriors, forced_identifiability_transform


def test_forced_identifiability_transform():
    """Test that the transform produces sorted values."""
    print("\nTesting forced identifiability transform...")
    
    # Test with random uniform samples
    key = jax.random.PRNGKey(42)
    n_samples = 1000
    n_peaks = 3
    
    for _ in range(5):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=(n_peaks,))
        
        # Apply transform
        t = forced_identifiability_transform(x)
        
        # Check that values are sorted
        assert jnp.all(t[:-1] <= t[1:]), f"Transform did not produce sorted values: {t}"
        
        print(f"  Input:  {x}")
        print(f"  Output: {t} (sorted: {jnp.all(t[:-1] <= t[1:])})")
    
    print("  ✓ Transform produces sorted values")


def test_sorted_prior_sampling():
    """Test that sorted priors produce sorted arrival times."""
    print("\nTesting sorted prior sampling...")
    
    # Create priors with sorted_u=True
    priors = FRBPriors(
        model_name="emg",
        max_peaks=3,
        fit_pulses=False,
        sorted_u=True
    )
    
    # Sample from prior
    key = jax.random.PRNGKey(123)
    n_samples = 100
    samples = priors.sample_from_prior(key, n_samples)
    
    # Extract arrival times (indices 6, 7, 8 for 3-peak EMG)
    # Parameters are: A1, A2, A3, tau1, tau2, tau3, u1, u2, u3, w1, w2, w3, sigma
    u1 = samples[:, 6]
    u2 = samples[:, 7]
    u3 = samples[:, 8]
    
    # Check that u values are sorted for each sample
    sorted_correctly = jnp.all((u1 <= u2) & (u2 <= u3))
    
    print(f"  Sampled {n_samples} parameter sets")
    print(f"  All u values sorted: {sorted_correctly}")
    print(f"  Example u values: u1={u1[0]:.3f}, u2={u2[0]:.3f}, u3={u3[0]:.3f}")
    
    assert sorted_correctly, "Not all samples have sorted u values!"
    print("  ✓ Sorted priors produce sorted arrival times")
    
    # Compare with unsorted priors
    priors_unsorted = FRBPriors(
        model_name="emg",
        max_peaks=3,
        fit_pulses=False,
        sorted_u=False
    )
    
    key = jax.random.PRNGKey(456)
    samples_unsorted = priors_unsorted.sample_from_prior(key, n_samples)
    
    u1_unsorted = samples_unsorted[:, 6]
    u2_unsorted = samples_unsorted[:, 7]
    u3_unsorted = samples_unsorted[:, 8]
    
    # Check that unsorted priors don't necessarily produce sorted values
    unsorted_violations = jnp.sum((u1_unsorted > u2_unsorted) | (u2_unsorted > u3_unsorted))
    
    print(f"\n  Unsorted priors: {unsorted_violations}/{n_samples} samples have unsorted u values")
    print(f"  Example unsorted: u1={u1_unsorted[0]:.3f}, u2={u2_unsorted[0]:.3f}, u3={u3_unsorted[0]:.3f}")


def test_parameter_ranges():
    """Test that sorted u values respect the prior bounds."""
    print("\nTesting parameter ranges with sorted priors...")
    
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 10.0},
        'tau': {'min': 0.001, 'max': 10.0},
        'u': {'min': -2.0, 'max': 5.0},  # Custom range
        'width': {'min': 0.001, 'max': 5.0},
        'baseline': {'min': -1.0, 'max': 1.0},
        'log_sigma': {'min': jnp.log(0.0001), 'max': jnp.log(2.0)},
    }
    
    priors = FRBPriors(
        model_name="emg",
        max_peaks=4,
        fit_pulses=False,
        prior_bounds=prior_bounds,
        sorted_u=True
    )
    
    # Sample from prior
    key = jax.random.PRNGKey(789)
    n_samples = 500
    samples = priors.sample_from_prior(key, n_samples)
    
    # Extract u values (indices 8-11 for 4-peak EMG)
    u_values = samples[:, 8:12]
    
    # Check bounds
    u_min = prior_bounds['u']['min']
    u_max = prior_bounds['u']['max']
    
    in_bounds = jnp.all((u_values >= u_min) & (u_values <= u_max))
    sorted_correctly = jnp.all(u_values[:, :-1] <= u_values[:, 1:])
    
    print(f"  u range: [{u_min}, {u_max}]")
    print(f"  All u values in bounds: {in_bounds}")
    print(f"  All u values sorted: {sorted_correctly}")
    print(f"  Min u sampled: {jnp.min(u_values):.3f}")
    print(f"  Max u sampled: {jnp.max(u_values):.3f}")
    
    assert in_bounds, "Some u values are out of bounds!"
    assert sorted_correctly, "Some u values are not sorted!"
    print("  ✓ Sorted u values respect prior bounds")


if __name__ == "__main__":
    print("="*60)
    print("TESTING SORTED PRIORS FOR ARRIVAL TIMES")
    print("="*60)
    
    test_forced_identifiability_transform()
    test_sorted_prior_sampling()
    test_parameter_ranges()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)