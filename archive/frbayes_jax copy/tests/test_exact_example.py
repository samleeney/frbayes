"""
Exact replication of the BlackJAX example to verify it works.
"""
import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise

# Set random seed
rng_key = jax.random.PRNGKey(0)

# Define likelihood and prior EXACTLY as in the example
loglikelihood_function = lambda x: jax.scipy.stats.multivariate_normal.logpdf(
    x, jnp.ones(5), jnp.eye(5)*0.01
)
logprior_function = lambda x: jax.scipy.stats.norm.logpdf(x).sum()

# Initialize nested sampling algorithm
algo = blackjax.nss(
    logprior_fn=logprior_function,
    loglikelihood_fn=loglikelihood_function,
    num_delete=50,
    num_inner_steps=20,
)

# Initialize state with live points
rng_key, sampling_key, initialization_key = jax.random.split(rng_key, 3)
initial_live_points = jax.random.normal(initialization_key, (1000, 5))
print(f"Initial points shape: {initial_live_points.shape}")

print("Initializing state...")
state = algo.init(initial_live_points)
print(f"Initial logZ: {state.logZ}")

# Take ONE step without JIT first
print("Taking one step without JIT...")
rng_key, step_key = jax.random.split(rng_key)
state, dead = algo.step(step_key, state)
print(f"Step completed! New logZ: {state.logZ}")

print("Success! The exact example works.")