"""
BlackJAX nested sampling integration for FRBayes using distrax priors.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap
import blackjax
from blackjax.ns.utils import finalise
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import tqdm
from .models import get_model_function, get_num_params, get_sigma_index, get_param_names
from .priors import FRBPriors


def run_nested_sampling(
    model_name: str,
    data: np.ndarray,
    t: np.ndarray,
    prior_bounds: Optional[Dict] = None,
    max_peaks: int = 2,
    fit_pulses: bool = False,
    num_live_points: int = 1000,
    num_delete: int = 50,
    num_inner_steps: int = 20,
    log_tolerance: float = -3.0,
    seed: int = 0
):
    """
    Run nested sampling using BlackJAX with distrax priors.
    
    Args:
        model_name: Name of the model to use
        data: Observed pulse profile
        t: Time axis
        prior_bounds: Prior bounds for parameters (uses defaults if None)
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
        num_live_points: Number of live points
        num_delete: Number of points to delete per iteration
        num_inner_steps: Number of MCMC steps between replacements
        log_tolerance: Termination criterion (log(Z_live/Z))
        seed: Random seed
        verbose: Whether to print progress
        max_iterations: Maximum number of iterations
    
    Returns:
        Dictionary containing sampling results
    """
    # Convert data to JAX arrays
    data_jax = jnp.array(data)
    t_jax = jnp.array(t)
    
    # Initialize prior system
    priors = FRBPriors(model_name, max_peaks, fit_pulses, prior_bounds)
    ndims = priors.ndims
    
    # Get model function
    model_func = get_model_function(model_name)
    sigma_idx = get_sigma_index(model_name, max_peaks, fit_pulses)
    
    # Create log-likelihood function
    def loglikelihood_fn(theta):
        # Get model prediction
        model_pred = model_func(t_jax, theta, max_peaks, fit_pulses)
        
        # Get sigma
        sigma = theta[sigma_idx]
        
        # Calculate residuals
        residuals = data_jax - model_pred
        
        # Gaussian log-likelihood
        n = len(data_jax)
        log_likelihood = -0.5 * jnp.sum((residuals / sigma) ** 2) - n * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
        
        return log_likelihood
    
    # Create distrax distributions for the prior
    import distrax
    bounds = priors.prior_bounds
    
    # Build list of distributions
    dists = []
    
    # Amplitudes - uniform
    for i in range(max_peaks):
        dists.append(distrax.Uniform(
            low=bounds['amplitude']['min'],
            high=bounds['amplitude']['max']
        ))
    
    # Taus - uniform  
    for i in range(max_peaks):
        dists.append(distrax.Uniform(
            low=bounds['tau']['min'],
            high=bounds['tau']['max']
        ))
    
    # Arrival times - uniform
    for i in range(max_peaks):
        dists.append(distrax.Uniform(
            low=bounds['u']['min'],
            high=bounds['u']['max']
        ))
    
    # Widths (for EMG) - uniform
    if 'emg' in model_name:
        for i in range(max_peaks):
            dists.append(distrax.Uniform(
                low=bounds['width']['min'],
                high=bounds['width']['max']
            ))
    
    # Baseline (if applicable) - uniform
    if 'baseline' in model_name:
        dists.append(distrax.Uniform(
            low=bounds['baseline']['min'],
            high=bounds['baseline']['max']
        ))
    
    # Sigma - just uniform (not log-uniform for simplicity)
    dists.append(distrax.Uniform(
        low=jnp.exp(bounds['log_sigma']['min']),
        high=jnp.exp(bounds['log_sigma']['max'])
    ))
    
    # Npulse (if fitted) - uniform over integer range
    if fit_pulses:
        dists.append(distrax.Uniform(
            low=1.0,
            high=float(max_peaks)
        ))
    
    @jit
    def logprior_fn(theta):
        # Simple: sum log probabilities from each distribution
        # Access each parameter directly like in the supernova example
        logp = jnp.sum(jnp.array([dists[i].log_prob(theta[i]) for i in range(len(dists))]))
        return logp
    
    # Initialize nested sampling algorithm
    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )
    
    # Initialize random key
    rng_key = jax.random.PRNGKey(seed)
    
    # Sample initial points from the prior using distrax
    rng_key, init_key = jax.random.split(rng_key)
    initial_live_points = priors.sample_from_prior(init_key, num_live_points)
    
    # Initialize state
    state = algo.init(initial_live_points)
    
    # JIT-compile the step function
    @jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point
    
    # Run nested sampling until convergence
    dead = []
    pbar = tqdm.tqdm(desc="Dead points", unit=" dead points")
    
    iteration = 0
    while state.logZ_live - state.logZ >= log_tolerance:
        # Take a step
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(num_delete)
        
        iteration += 1
        
        # Update progress bar every 100 iterations
        if iteration % 100 == 0:
            pbar.set_postfix({"logZ": f"{state.logZ:.2f}", "logZ_live": f"{state.logZ_live:.2f}"})
    
    pbar.close()
    
    # Finalize results - EXACTLY like in the example
    final_state = finalise(state, dead)
    
    # Fix NaN birth likelihoods from BlackJAX (issue with final live points)
    # Following standard nested sampling practice, set NaN values to max birth likelihood
    import numpy as np
    logL_birth = np.array(final_state.loglikelihood_birth)
    nan_mask = np.isnan(logL_birth)
    if np.any(nan_mask):
        max_valid = np.nanmax(logL_birth)
        logL_birth[nan_mask] = max_valid
        # Replace in the final_state
        final_state = final_state._replace(loglikelihood_birth=logL_birth)
    
    return final_state


