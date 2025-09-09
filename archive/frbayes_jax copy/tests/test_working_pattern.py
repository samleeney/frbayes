"""
Test using EXACTLY the pattern from the working example.
"""
import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise

# Set random seed
rng_key = jax.random.PRNGKey(0)

# Simple 2-parameter model for testing
# Parameters: [amplitude, sigma]

# Define likelihood - Gaussian around true model
true_amplitude = 1.0
true_sigma = 0.05
t_data = jnp.linspace(0, 4, 50)
true_model = true_amplitude * jnp.exp(-t_data)
data = true_model + jax.random.normal(jax.random.PRNGKey(1), true_model.shape) * true_sigma

def loglikelihood_function(x):
    amplitude, sigma = x
    model = amplitude * jnp.exp(-t_data)
    residuals = data - model
    n = len(data)
    return -0.5 * jnp.sum((residuals / sigma) ** 2) - n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

# Define prior - normal distributions for both parameters
# This matches the example's use of norm.logpdf
def logprior_function(x):
    amplitude, sigma = x
    # Normal prior for amplitude centered at 1 with std 0.5
    amp_prior = jax.scipy.stats.norm.logpdf(amplitude, loc=1.0, scale=0.5)
    # Log-normal prior for sigma (normal prior on log(sigma))
    sigma_prior = jax.scipy.stats.norm.logpdf(jnp.log(sigma), loc=jnp.log(0.05), scale=0.5)
    return amp_prior + sigma_prior

# Initialize nested sampling algorithm
algo = blackjax.nss(
    logprior_fn=logprior_function,
    loglikelihood_fn=loglikelihood_function,
    num_delete=10,
    num_inner_steps=10,
)

# Initialize state with live points
# Draw from the prior distribution
rng_key, sampling_key, initialization_key = jax.random.split(rng_key, 3)
# Sample amplitude from normal(1, 0.5)
amp_samples = jax.random.normal(initialization_key, (100, 1)) * 0.5 + 1.0
# Sample log(sigma) from normal(log(0.05), 0.5) then exponentiate
rng_key, sigma_key = jax.random.split(rng_key)
log_sigma_samples = jax.random.normal(sigma_key, (100, 1)) * 0.5 + jnp.log(0.05)
sigma_samples = jnp.exp(log_sigma_samples)
# Combine
initial_live_points = jnp.concatenate([amp_samples, sigma_samples], axis=1)

print(f"Initial points shape: {initial_live_points.shape}")
print(f"Initial amplitude range: [{initial_live_points[:, 0].min():.3f}, {initial_live_points[:, 0].max():.3f}]")
print(f"Initial sigma range: [{initial_live_points[:, 1].min():.4f}, {initial_live_points[:, 1].max():.4f}]")

state = algo.init(initial_live_points)
print(f"Initial logZ: {state.logZ}")

# JIT-compile the step function - EXACTLY as in the example
@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Try one step
print("\nTaking one step...")
(state, rng_key), dead_info = one_step((state, rng_key), None)
print(f"Success! New logZ: {state.logZ}")

print("\nThe pattern works!")