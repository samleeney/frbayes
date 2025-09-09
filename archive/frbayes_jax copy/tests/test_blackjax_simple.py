"""
Test BlackJAX nested sampling with a simple bounded uniform prior.
"""
import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise
import numpy as np


def main():
    print("Testing BlackJAX with bounded uniform priors...")
    
    # Simple 2D uniform box
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    
    def logprior(x):
        # Uniform prior within bounds
        within_bounds = jnp.all((x >= bounds[:, 0]) & (x <= bounds[:, 1]))
        return jnp.where(within_bounds, 0.0, -jnp.inf)
    
    def loglikelihood(x):
        # Simple Gaussian likelihood centered at (1, 1)
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, jnp.array([1.0, 1.0]), jnp.eye(2) * 0.1
        )
    
    # Create algorithm
    algo = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood,
        num_delete=5,
        num_inner_steps=10,
    )
    
    # Initialize with points uniformly distributed in the box
    rng_key = jax.random.PRNGKey(0)
    initial_points = jax.random.uniform(
        rng_key, (20, 2), minval=bounds[:, 0], maxval=bounds[:, 1]
    )
    
    print(f"Initial points shape: {initial_points.shape}")
    print(f"Initial points range: [{initial_points.min():.2f}, {initial_points.max():.2f}]")
    
    # Initialize state
    print("\nInitializing state...")
    state = algo.init(initial_points)
    print(f"Initial logZ: {state.logZ}")
    
    # Take a few steps
    print("\nTaking steps...")
    for i in range(5):
        rng_key, subkey = jax.random.split(rng_key)
        state, dead = algo.step(subkey, state)
        print(f"Step {i+1}: logZ = {state.logZ:.3f}")
    
    print("\nSuccess! BlackJAX is working with bounded uniform priors.")
    
    # Now test with FRB-like parameter space
    print("\n" + "="*50)
    print("Testing with FRB-like parameter space...")
    
    # 3 parameters: amplitude, tau, sigma
    bounds_frb = jnp.array([
        [0.1, 2.0],    # amplitude
        [0.1, 1.0],    # tau  
        [0.01, 0.2]    # sigma
    ])
    
    def logprior_frb(x):
        within_bounds = jnp.all((x >= bounds_frb[:, 0]) & (x <= bounds_frb[:, 1]))
        # Also check sigma > 0 explicitly
        sigma_positive = x[2] > 0
        return jnp.where(within_bounds & sigma_positive, 0.0, -jnp.inf)
    
    # Simple exponential pulse model
    def simple_model(t, amplitude, tau):
        return amplitude * jnp.exp(-t / tau)
    
    # Generate synthetic data
    t_data = jnp.linspace(0, 4, 50)
    true_amplitude = 1.0
    true_tau = 0.5
    true_sigma = 0.05
    true_model = simple_model(t_data, true_amplitude, true_tau)
    data = true_model + jax.random.normal(jax.random.PRNGKey(1), true_model.shape) * true_sigma
    
    def loglikelihood_frb(x):
        amplitude, tau, sigma = x
        model = simple_model(t_data, amplitude, tau)
        residuals = data - model
        n = len(data)
        return -0.5 * jnp.sum((residuals / sigma) ** 2) - n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    
    # Create algorithm
    algo_frb = blackjax.nss(
        logprior_fn=logprior_frb,
        loglikelihood_fn=loglikelihood_frb,
        num_delete=5,
        num_inner_steps=10,
    )
    
    # Initialize with points uniformly distributed in the box
    rng_key = jax.random.PRNGKey(2)
    initial_points_frb = jax.random.uniform(
        rng_key, (20, 3), minval=bounds_frb[:, 0], maxval=bounds_frb[:, 1]
    )
    
    print(f"\nInitial FRB points shape: {initial_points_frb.shape}")
    
    # Initialize state
    print("Initializing FRB state...")
    state_frb = algo_frb.init(initial_points_frb)
    print(f"Initial logZ: {state_frb.logZ}")
    
    # Take a few steps
    print("\nTaking FRB steps...")
    for i in range(5):
        rng_key, subkey = jax.random.split(rng_key)
        state_frb, dead_frb = algo_frb.step(subkey, state_frb)
        print(f"Step {i+1}: logZ = {state_frb.logZ:.3f}")
    
    print("\nSuccess! BlackJAX works with FRB-like model.")


if __name__ == "__main__":
    main()