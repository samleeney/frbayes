"""
Diagnostic test to identify the hanging issue.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable JIT for debugging
jax.config.update("jax_disable_jit", True)

from frbayes_jax.models import emg_model, get_model_function
from frbayes_jax.priors import transform_prior_emg
from frbayes_jax.utils import get_default_prior_ranges


def main():
    """
    Diagnostic test.
    """
    print("="*60)
    print("DIAGNOSTIC TEST")
    print("="*60)
    
    # Settings
    model_name = "emg"
    max_peaks = 2
    fit_pulses = False
    
    # Test 1: Model function
    print("\n1. Testing model function...")
    t = jnp.linspace(0, 4, 100)
    theta = jnp.array([0.8, 0.5, 0.5, 0.3, 1.0, 2.5, 0.15, 0.12, 0.05])
    
    model_func = get_model_function(model_name)
    start = time.time()
    result = model_func(t, theta, max_peaks, fit_pulses)
    print(f"   Model evaluation took {time.time() - start:.3f}s")
    print(f"   Result shape: {result.shape}, range: [{float(result.min()):.3f}, {float(result.max()):.3f}]")
    
    # Test 2: Prior transform
    print("\n2. Testing prior transform...")
    prior_ranges = get_default_prior_ranges(model_name)
    hypercube = jnp.array([0.5] * 9)  # Middle of unit hypercube
    
    start = time.time()
    theta_transformed = transform_prior_emg(hypercube, prior_ranges, max_peaks, fit_pulses)
    print(f"   Prior transform took {time.time() - start:.3f}s")
    print(f"   Transformed shape: {theta_transformed.shape}")
    
    # Test 3: Likelihood evaluation
    print("\n3. Testing likelihood...")
    data = result + jax.random.normal(jax.random.PRNGKey(0), result.shape) * 0.05
    
    def loglikelihood(theta):
        model_pred = model_func(t, theta, max_peaks, fit_pulses)
        sigma = theta[-1]
        residuals = data - model_pred
        n = len(data)
        return -0.5 * jnp.sum((residuals / sigma) ** 2) - n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    
    start = time.time()
    logL = loglikelihood(theta)
    print(f"   Likelihood evaluation took {time.time() - start:.3f}s")
    print(f"   Log-likelihood: {float(logL):.3f}")
    
    # Test 4: BlackJAX import
    print("\n4. Testing BlackJAX import...")
    try:
        import blackjax
        from blackjax.ns.utils import finalise
        print("   BlackJAX imported successfully")
        
        # Test algorithm creation
        def logprior(x):
            return 0.0
        
        algo = blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=loglikelihood,
            num_delete=10,
            num_inner_steps=5,
        )
        print("   Algorithm created successfully")
        
        # Test initialization with small number of points
        print("\n5. Testing BlackJAX initialization...")
        initial_points = jax.random.normal(jax.random.PRNGKey(0), (50, 9))
        
        start = time.time()
        state = algo.init(initial_points)
        print(f"   Initialization took {time.time() - start:.3f}s")
        print(f"   Initial logZ: {float(state.logZ):.3f}")
        
        # Test single step
        print("\n6. Testing single BlackJAX step...")
        rng_key = jax.random.PRNGKey(1)
        
        start = time.time()
        state, dead_point = algo.step(rng_key, state)
        print(f"   Single step took {time.time() - start:.3f}s")
        print(f"   Updated logZ: {float(state.logZ):.3f}")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()