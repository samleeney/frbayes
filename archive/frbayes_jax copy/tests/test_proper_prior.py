"""
Test with proper uniform prior that returns actual log probabilities.
"""
import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise
import numpy as np


def main():
    print("Testing BlackJAX with proper uniform prior...")
    
    # Define bounds
    lower = jnp.array([-5.0, -5.0])
    upper = jnp.array([5.0, 5.0])
    
    def logprior(x):
        # Proper uniform prior: log(1/volume) if in bounds, -inf otherwise
        within_bounds = jnp.all((x >= lower) & (x <= upper))
        volume = jnp.prod(upper - lower)
        log_prob = -jnp.log(volume)  # log(1/volume)
        return jnp.where(within_bounds, log_prob, -jnp.inf)
    
    def loglikelihood(x):
        # Simple Gaussian likelihood
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, jnp.array([1.0, 1.0]), jnp.eye(2) * 0.1
        )
    
    # Create algorithm FIRST (before generating initial points)
    algo = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood,
        num_delete=5,
        num_inner_steps=10,
    )
    
    # Generate initial points from uniform distribution within bounds
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_points = jax.random.uniform(
        init_key, (20, 2), minval=lower, maxval=upper
    )
    
    print(f"Initial points shape: {initial_points.shape}")
    print(f"First point: {initial_points[0]}")
    print(f"Log prior of first point: {logprior(initial_points[0]):.3f}")
    print(f"Log likelihood of first point: {loglikelihood(initial_points[0]):.3f}")
    
    # Initialize state
    print("\nInitializing state...")
    state = algo.init(initial_points)
    print(f"Initial logZ: {state.logZ}")
    
    # Take one step without JIT to see what happens
    print("\nTaking one step (no JIT)...")
    rng_key, step_key = jax.random.split(rng_key)
    state, dead = algo.step(step_key, state)
    print(f"Step completed! New logZ: {state.logZ:.3f}")
    
    print("\nSuccess!")


if __name__ == "__main__":
    main()