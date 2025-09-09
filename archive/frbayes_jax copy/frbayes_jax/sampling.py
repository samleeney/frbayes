"""
Fixed BlackJAX nested sampling integration for FRBayes.
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
from .priors import get_prior_transform


def create_logprior_uniform(bounds: Dict[str, Tuple[float, float]], param_indices: Dict[str, Tuple[int, int]]) -> Callable:
    """
    Create a log-prior function for uniform distributions.
    
    Args:
        bounds: Dictionary of parameter bounds
        param_indices: Dictionary mapping parameter names to index ranges
    
    Returns:
        Log-prior function that returns actual log probabilities
    """
    def logprior(theta):
        log_prob = 0.0
        
        # Check each parameter group
        for param_name, (idx_start, idx_end) in param_indices.items():
            if param_name in bounds:
                param_bounds = bounds[param_name]
                params = theta[idx_start:idx_end] if idx_end > idx_start + 1 else theta[idx_start]
                
                # Check bounds
                if idx_end > idx_start + 1:
                    # Multiple parameters
                    within = jnp.all((params >= param_bounds["min"]) & (params <= param_bounds["max"]))
                else:
                    # Single parameter
                    within = (params >= param_bounds["min"]) & (params <= param_bounds["max"])
                
                # Add log probability (log(1/width) for uniform)
                width = param_bounds["max"] - param_bounds["min"]
                log_prob = jnp.where(within, log_prob - jnp.log(width), -jnp.inf)
        
        return log_prob
    
    return logprior


def sample_from_prior_uniform(
    rng_key: jax.random.PRNGKey,
    bounds: Dict[str, Tuple[float, float]],
    param_indices: Dict[str, Tuple[int, int]],
    ndims: int,
    nsamples: int
) -> jnp.ndarray:
    """
    Sample from uniform prior distributions.
    
    Args:
        rng_key: Random key
        bounds: Dictionary of parameter bounds
        param_indices: Dictionary mapping parameter names to index ranges
        ndims: Total number of dimensions
        nsamples: Number of samples
    
    Returns:
        Array of samples from the prior
    """
    samples = jnp.zeros((nsamples, ndims))
    
    for param_name, (idx_start, idx_end) in param_indices.items():
        if param_name in bounds:
            param_bounds = bounds[param_name]
            rng_key, subkey = jax.random.split(rng_key)
            
            n_params = idx_end - idx_start if idx_end > idx_start + 1 else 1
            param_samples = jax.random.uniform(
                subkey, 
                (nsamples, n_params),
                minval=param_bounds["min"],
                maxval=param_bounds["max"]
            )
            
            if n_params == 1:
                samples = samples.at[:, idx_start].set(param_samples.squeeze())
            else:
                samples = samples.at[:, idx_start:idx_end].set(param_samples)
    
    # Special handling for sorted arrival times
    if "u" in param_indices:
        idx_start, idx_end = param_indices["u"]
        # Sort each sample's arrival times
        for i in range(nsamples):
            samples = samples.at[i, idx_start:idx_end].set(
                jnp.sort(samples[i, idx_start:idx_end])
            )
    
    return samples


def run_nested_sampling(
    model_name: str,
    data: np.ndarray,
    t: np.ndarray,
    prior_ranges: Dict,
    max_peaks: int,
    fit_pulses: bool,
    num_live_points: int = 1000,
    num_delete: int = 50,
    num_inner_steps: int = 20,
    max_iterations: int = 10000,
    log_tolerance: float = -3.0,
    seed: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Run nested sampling using BlackJAX with proper prior handling.
    
    Args:
        model_name: Name of the model to use
        data: Observed pulse profile
        t: Time axis
        prior_ranges: Prior ranges for parameters
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
        num_live_points: Number of live points
        num_delete: Number of points to delete per iteration
        num_inner_steps: Number of MCMC steps between replacements
        max_iterations: Maximum number of iterations
        log_tolerance: Termination criterion (log(Z_live/Z))
        seed: Random seed
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary containing sampling results
    """
    # Convert data to JAX arrays
    data_jax = jnp.array(data)
    t_jax = jnp.array(t)
    
    # Get model function and dimensions
    model_func = get_model_function(model_name)
    ndims = get_num_params(model_name, max_peaks, fit_pulses)
    sigma_idx = get_sigma_index(model_name, max_peaks, fit_pulses)
    
    # Set up parameter indices for proper prior handling
    param_indices = {}
    idx = 0
    
    # Amplitudes
    param_indices["amplitude"] = (0, max_peaks)
    idx = max_peaks
    
    # Tau values
    param_indices["tau"] = (idx, idx + max_peaks)
    idx += max_peaks
    
    # Arrival times
    param_indices["u"] = (idx, idx + max_peaks)
    idx += max_peaks
    
    # Width parameters (for EMG)
    if "emg" in model_name:
        param_indices["width"] = (idx, idx + max_peaks)
        idx += max_peaks
    
    # Baseline (if applicable)
    if "baseline" in model_name:
        param_indices["baseline_offset"] = (idx, idx + 1)
        idx += 1
    
    # Sigma
    param_indices["sigma"] = (sigma_idx, sigma_idx + 1)
    
    # Npulse (if fitted)
    if fit_pulses:
        param_indices["npulse"] = (ndims - 1, ndims)
        # Add npulse to prior_ranges if not there
        if "npulse" not in prior_ranges:
            prior_ranges["npulse"] = {"min": 1, "max": max_peaks}
    
    # Create log-prior function
    logprior_fn = create_logprior_uniform(prior_ranges, param_indices)
    
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
    
    # Initialize nested sampling algorithm
    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )
    
    # Initialize random key
    rng_key = jax.random.PRNGKey(seed)
    
    # Sample initial points from the prior
    rng_key, init_key = jax.random.split(rng_key)
    initial_live_points = sample_from_prior_uniform(
        init_key, prior_ranges, param_indices, ndims, num_live_points
    )
    
    if verbose:
        print(f"Initial points shape: {initial_live_points.shape}")
        print(f"Initial points sigma range: [{initial_live_points[:, sigma_idx].min():.4f}, {initial_live_points[:, sigma_idx].max():.4f}]")
    
    # Initialize state
    state = algo.init(initial_live_points)
    
    if verbose:
        print(f"Initial logZ: {state.logZ}")
    
    # JIT-compile the step function (following the example pattern exactly)
    @jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point
    
    # Run nested sampling
    dead = []
    
    if verbose:
        pbar = tqdm.tqdm(desc="Dead points", unit=" dead points")
    
    iteration = 0
    while iteration < max_iterations:
        # Check termination criterion
        if state.logZ_live - state.logZ < log_tolerance:
            if verbose:
                print(f"\nConverged: logZ_live - logZ = {state.logZ_live - state.logZ:.4f}")
            break
        
        # Take a step
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        
        if verbose:
            pbar.update(num_delete)
        
        iteration += 1
        
        # Print progress every 100 iterations
        if verbose and iteration % 100 == 0:
            pbar.set_postfix({"logZ": f"{state.logZ:.2f}", "logZ_live": f"{state.logZ_live:.2f}"})
    
    if verbose:
        pbar.close()
        print(f"Completed {iteration} iterations")
        print(f"Final log(Z) = {state.logZ:.4f}")
    
    # Finalize results
    final_info = finalise(state, dead)
    
    # Convert results to numpy for compatibility with analysis tools
    results = {
        'particles': np.array(final_info.particles),
        'logL': np.array(final_info.loglikelihood),
        'logL_birth': np.array(final_info.loglikelihood_birth),
        'logZ': float(state.logZ),  # Use state's logZ
        'logZ_error': 0.0,  # Not available in this version
        'ndims': ndims,
        'nsamples': len(final_info.particles),
        'model_name': model_name,
        'max_peaks': max_peaks,
        'fit_pulses': fit_pulses
    }
    
    return results


def save_chains_for_anesthetic(results: Dict, filename: str):
    """
    Save nested sampling results in anesthetic-compatible format.
    
    Args:
        results: Dictionary containing sampling results
        filename: Output filename (will append _dead-birth.txt)
    """
    # Stack data: parameters, death likelihood, birth likelihood
    data = np.column_stack([
        results['particles'],
        results['logL'],
        results['logL_birth']
    ])
    
    # Save in space-separated format
    np.savetxt(f"{filename}_dead-birth.txt", data)
    
    # Also save basic stats file
    with open(f"{filename}.stats", 'w') as f:
        f.write(f"log(Z) = {results['logZ']:.6f}\n")
        f.write(f"log(Z) error = {results['logZ_error']:.6f}\n")
        f.write(f"Number of samples = {results['nsamples']}\n")
        f.write(f"Number of dimensions = {results['ndims']}\n")