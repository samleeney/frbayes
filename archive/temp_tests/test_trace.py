"""
Trace exact location of hanging.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise

# Disable JIT
jax.config.update("jax_disable_jit", True)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import emg_model, get_model_function
from frbayes_jax.priors import transform_prior_emg
from frbayes_jax.utils import get_default_prior_ranges


def main():
    print("Testing BlackJAX step directly...")
    
    # Simple 2D Gaussian test
    print("\n1. Testing with simple Gaussian...")
    
    def loglikelihood(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, jnp.ones(2), jnp.eye(2)*0.01)
    
    def logprior(x):
        return jax.scipy.stats.norm.logpdf(x).sum()
    
    # Create algorithm
    algo = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood,
        num_delete=5,
        num_inner_steps=2,
    )
    
    # Initialize
    rng_key = jax.random.PRNGKey(0)
    initial_points = jax.random.normal(rng_key, (20, 2))
    
    print("   Initializing state...")
    state = algo.init(initial_points)
    print(f"   Initial logZ: {state.logZ}")
    
    print("   Taking one step...")
    rng_key, subkey = jax.random.split(rng_key)
    
    try:
        state, dead = algo.step(subkey, state)
        print(f"   Step completed! New logZ: {state.logZ}")
    except Exception as e:
        print(f"   Error in step: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test with FRB model
    print("\n2. Testing with FRB model...")
    
    # Settings
    model_name = "emg"
    max_peaks = 2
    fit_pulses = False
    
    # Generate simple data
    t = jnp.linspace(0, 4, 50)
    theta_true = jnp.array([0.8, 0.5, 0.5, 0.3, 1.0, 2.5, 0.15, 0.12, 0.05])
    
    model_func = get_model_function(model_name)
    data = model_func(t, theta_true, max_peaks, fit_pulses)
    data = data + jax.random.normal(jax.random.PRNGKey(1), data.shape) * 0.05
    
    def frb_loglikelihood(theta):
        model_pred = model_func(t, theta, max_peaks, fit_pulses)
        sigma = theta[8]  # sigma index for EMG with 2 peaks
        residuals = data - model_pred
        n = len(data)
        return -0.5 * jnp.sum((residuals / sigma) ** 2) - n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    
    def frb_logprior(theta):
        sigma = theta[8]
        return jnp.where(sigma > 0, 0.0, -jnp.inf)
    
    # Create algorithm
    algo_frb = blackjax.nss(
        logprior_fn=frb_logprior,
        loglikelihood_fn=frb_loglikelihood,
        num_delete=5,
        num_inner_steps=2,
    )
    
    # Generate initial points
    prior_ranges = get_default_prior_ranges(model_name)
    prior_ranges["amplitude"]["max"] = 2.0
    prior_ranges["u"]["max"] = 4.0
    
    hypercube = jax.random.uniform(jax.random.PRNGKey(2), (20, 9))
    from jax import vmap
    initial_points_frb = vmap(lambda x: transform_prior_emg(x, prior_ranges, max_peaks, fit_pulses))(hypercube)
    
    print("   Initializing FRB state...")
    state_frb = algo_frb.init(initial_points_frb)
    print(f"   Initial logZ: {state_frb.logZ}")
    
    print("   Taking one FRB step...")
    rng_key, subkey = jax.random.split(rng_key)
    
    try:
        state_frb, dead_frb = algo_frb.step(subkey, state_frb)
        print(f"   Step completed! New logZ: {state_frb.logZ}")
    except Exception as e:
        print(f"   Error in step: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()