"""
BlackJAX nested sampling integration for FRBayes using distrax priors.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap
import blackjax
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import tqdm
from .models import get_model_function, get_num_params, get_sigma_index, get_param_names, get_spectral_index_location
from .priors import FRBPriors, FRBPriors3D
from blackjax.ns.base import NSInfo


def finalise_chunked(state, dead, chunk_size=10):
    """
    Memory-efficient version of BlackJAX's finalise function.
    Processes dead points in smaller batches to avoid memory issues.
    
    Args:
        state: Final NSState with live particles
        dead: List of NSInfo objects from dead points
        chunk_size: Number of dead points to concatenate at once
    
    Returns:
        Combined NSInfo object with all dead and live points
    """
    import jax
    import jax.numpy as jnp
    import gc
    
    # Handle edge case: no dead points
    if len(dead) == 0:
        return NSInfo(
            state.particles,
            state.loglikelihood,
            state.loglikelihood_birth,
            state.logprior,
            None  # No inner kernel info
        )
    
    # Process dead points in smaller chunks to avoid memory spikes
    # First, concatenate dead points in small groups
    processed_chunks = []
    
    for i in range(0, len(dead), chunk_size):
        chunk = dead[i:min(i + chunk_size, len(dead))]
        
        if len(chunk) == 1:
            processed_chunks.append(chunk[0])
        else:
            # Concatenate within this small chunk
            # Use regular concatenate but with small number of arrays
            chunk_particles = jnp.concatenate([d.particles for d in chunk], axis=0)
            chunk_logL = jnp.concatenate([d.loglikelihood for d in chunk], axis=0)
            chunk_logL_birth = jnp.concatenate([d.loglikelihood_birth for d in chunk], axis=0)
            chunk_logprior = jnp.concatenate([d.logprior for d in chunk], axis=0)
            
            processed_chunks.append(NSInfo(
                chunk_particles,
                chunk_logL,
                chunk_logL_birth,
                chunk_logprior,
                chunk[-1].inner_kernel_info  # Keep last kernel info
            ))
        
        # Explicitly clear memory after each chunk
        if i % (chunk_size * 10) == 0:
            gc.collect()
    
    # Now we have fewer, larger chunks to combine
    # Use iterative pairwise merging to avoid memory spikes
    while len(processed_chunks) > 1:
        next_level = []
        
        for i in range(0, len(processed_chunks), 2):
            if i + 1 < len(processed_chunks):
                # Merge pair
                left = processed_chunks[i]
                right = processed_chunks[i + 1]
                
                merged = NSInfo(
                    jnp.concatenate([left.particles, right.particles], axis=0),
                    jnp.concatenate([left.loglikelihood, right.loglikelihood], axis=0),
                    jnp.concatenate([left.loglikelihood_birth, right.loglikelihood_birth], axis=0),
                    jnp.concatenate([left.logprior, right.logprior], axis=0),
                    right.inner_kernel_info  # Keep latest kernel info
                )
                next_level.append(merged)
            else:
                # Odd one out, just pass through
                next_level.append(processed_chunks[i])
        
        processed_chunks = next_level
        gc.collect()
    
    # Now add the final live points
    final_live = NSInfo(
        state.particles,
        state.loglikelihood,
        state.loglikelihood_birth,
        state.logprior,
        dead[-1].inner_kernel_info if len(dead) > 0 else None
    )
    
    # Final concatenation
    accumulated = processed_chunks[0] if processed_chunks else dead[0]
    
    return NSInfo(
        jnp.concatenate([accumulated.particles, final_live.particles], axis=0),
        jnp.concatenate([accumulated.loglikelihood, final_live.loglikelihood], axis=0),
        jnp.concatenate([accumulated.loglikelihood_birth, final_live.loglikelihood_birth], axis=0),
        jnp.concatenate([accumulated.logprior, final_live.logprior], axis=0),
        final_live.inner_kernel_info
    )


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
    
    Returns:
        Dictionary containing sampling results
    """
    # Convert data to JAX arrays
    data_jax = jnp.array(data)
    t_jax = jnp.array(t)
    
    # Initialize prior system (always uses sorted priors)
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
    
    # Arrival times/location parameters
    if 'periodic' in model_name:
        # u0 - first pulse location
        u0_bounds = bounds.get('u0', bounds.get('u', {'min': 0.0, 'max': 4.0}))
        dists.append(distrax.Uniform(
            low=u0_bounds['min'],
            high=u0_bounds['max']
        ))
        # period - spacing between pulses
        dists.append(distrax.Uniform(
            low=bounds['period']['min'],
            high=bounds['period']['max']
        ))
    else:
        # Regular models - will be handled specially in logprior_fn for sorting
        for i in range(max_peaks):
            dists.append(None)  # Handled specially for sorting constraint
    
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
        logp = 0.0
        
        # Amplitudes and Taus - regular priors
        for i in range(2 * max_peaks):
            if dists[i] is not None:
                logp += dists[i].log_prob(theta[i])
        
        if 'periodic' in model_name:
            # For periodic models: u0 and period have simple uniform priors
            u0_idx = 2 * max_peaks
            period_idx = 2 * max_peaks + 1
            
            # u0 and period priors
            if dists[u0_idx] is not None:
                logp += dists[u0_idx].log_prob(theta[u0_idx])
            if dists[period_idx] is not None:
                logp += dists[period_idx].log_prob(theta[period_idx])
            
            # Continue with remaining parameters
            start_idx = 2 * max_peaks + 2
        else:
            # Non-periodic: Arrival times - enforce sorting constraint
            u_start = 2 * max_peaks
            u_end = 3 * max_peaks
            u_values = theta[u_start:u_end]
            
            if max_peaks > 1:
                # Check if values are sorted
                is_sorted = jnp.all(u_values[:-1] <= u_values[1:])
                
                # If not sorted, return -inf (invalid)
                logp = jnp.where(is_sorted, logp, -jnp.inf)
            
            # Check bounds for u values
            u_min = bounds['u']['min']
            u_max = bounds['u']['max']
            in_bounds = jnp.all((u_values >= u_min) & (u_values <= u_max))
            logp = jnp.where(in_bounds, logp, -jnp.inf)
            
            # Uniform prior on sorted values
            logp += -max_peaks * jnp.log(u_max - u_min)
            
            start_idx = u_end
        
        # Continue with remaining parameters (widths, baseline, sigma, etc.)
        for i in range(start_idx, len(dists)):
            if dists[i] is not None:
                logp += dists[i].log_prob(theta[i])
        
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
    
    print("\n" + "="*60)
    print("SAMPLING COMPLETED - Starting post-processing")
    print("="*60)
    
    # Clear memory before finalise to prevent OOM errors with large dead point arrays
    import gc
    gc.collect()
    jax.clear_caches()  # Clear JAX's compiled function cache
    
    # Check memory status before finalise
    print("\nMemory status before finalise:")
    try:
        stats = jax.devices()[0].memory_stats()
        print(f"  Bytes in use: {stats['bytes_in_use'] / 1e9:.2f} GB")
        print(f"  Peak bytes: {stats['peak_bytes_in_use'] / 1e9:.2f} GB")
        print(f"  Bytes limit: {stats['bytes_limit'] / 1e9:.2f} GB")
        print(f"  Available: {(stats['bytes_limit'] - stats['bytes_in_use']) / 1e9:.2f} GB")
    except:
        pass
    
    print(f"\nNumber of dead points to concatenate: {len(dead)}")
    
    # Dynamic chunk size: divide total dead points into ~10 chunks
    chunk_size = max(1, len(dead) // 10)
    print(f"Using chunk size: {chunk_size} (processing in ~{len(dead) // chunk_size + 1} batches)")
    print(f"Attempting chunked finalise...")
    
    # Use our custom chunked finalise function
    # This processes dead points in batches to avoid memory issues
    final_state = finalise_chunked(state, dead, chunk_size=chunk_size)
    
    print("Chunked finalise completed successfully")
    
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


def run_nested_sampling_2d(
    model_name: str,
    data_2d: np.ndarray,
    t: np.ndarray,
    freq: np.ndarray,
    noise_per_channel: Optional[np.ndarray] = None,
    prior_bounds: Optional[Dict] = None,
    max_peaks: int = 2,
    fit_pulses: bool = False,
    num_live_points: int = 1000,
    num_delete: int = 50,
    num_inner_steps: int = 20,
    log_tolerance: float = -3.0,
    seed: int = 0,
    ref_freq: float = 1400.0,
    use_rfi_mitigation: bool = False
):
    """
    Run nested sampling for 2D models with spectral index.
    
    Args:
        model_name: Name of the 2D model to use (e.g., "emg_2d", "exponential_2d")
        data_2d: 2D waterfall data (freq, time)
        t: Time axis
        freq: Frequency axis (MHz)
        noise_per_channel: Per-channel noise estimates (if None, uses single sigma)
        prior_bounds: Prior bounds for parameters (uses defaults if None)
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
        num_live_points: Number of live points
        num_delete: Number of points to delete per iteration
        num_inner_steps: Number of MCMC steps between replacements
        log_tolerance: Termination criterion (log(Z_live/Z))
        seed: Random seed
        ref_freq: Reference frequency in MHz for spectral scaling
        use_rfi_mitigation: If True, uses Bayesian anomaly detection for RFI mitigation
    
    Returns:
        Dictionary containing sampling results
    """
    # Ensure we have a 2D model
    if "2d" not in model_name:
        raise ValueError(f"Model {model_name} is not a 2D model. Use models ending with '_2d'")
    
    # Convert data to JAX arrays
    data_2d_jax = jnp.array(data_2d)
    t_jax = jnp.array(t)
    freq_jax = jnp.array(freq)
    
    if noise_per_channel is not None:
        noise_jax = jnp.array(noise_per_channel)
    else:
        # Estimate single noise value from data
        noise_jax = jnp.std(data_2d_jax)
    
    # Initialize prior system with spectral index
    from .priors import FRBPriors
    priors = FRBPriors(model_name, max_peaks, fit_pulses, prior_bounds, use_rfi_mitigation)
    ndims = priors.ndims
    
    # Get model function and parameter indices
    model_func = get_model_function(model_name)
    sigma_idx = get_sigma_index(model_name, max_peaks, fit_pulses)
    alpha_idx = get_spectral_index_location(model_name, max_peaks)
    
    # Create log-likelihood function for 2D data
    if use_rfi_mitigation:
        # Calculate delta for anomaly detection (max intensity in the data)
        delta = jnp.max(jnp.abs(data_2d_jax))
        
        def loglikelihood_fn(theta):
            # Get 2D model prediction (exclude anomaly probability from model params)
            if use_rfi_mitigation:
                model_params = theta[:-1]  # All params except log_p
                log_p = theta[-1]  # Log anomaly probability
                p = jnp.exp(log_p)  # Anomaly probability
            else:
                model_params = theta
            
            model_2d = model_func(t_jax, freq_jax, model_params, max_peaks, fit_pulses, ref_freq)
            
            # Get sigma
            sigma = model_params[sigma_idx]
            
            # Calculate residuals
            residuals = data_2d_jax - model_2d
            
            # Calculate normal log-likelihood for each point
            if noise_per_channel is not None and len(noise_jax.shape) > 0:
                # Weight by per-channel noise
                weighted_residuals = residuals / noise_jax[:, None]
                log_likelihood_normal = -0.5 * weighted_residuals**2
                # Add normalization
                log_likelihood_normal -= jnp.log(noise_jax[:, None] * jnp.sqrt(2 * jnp.pi))
            else:
                # Use single sigma for all points
                log_likelihood_normal = -0.5 * (residuals / sigma) ** 2
                log_likelihood_normal -= jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
            
            # Apply anomaly correction (Bayesian anomaly detection)
            # Normal likelihood with prior probability (1-p)
            log_likelihood_with_prior = log_likelihood_normal + jnp.log(1 - p)
            
            # Anomaly threshold: uniform likelihood over [-delta, delta] with probability p
            anomaly_threshold = log_p - jnp.log(2 * delta)
            
            # Take maximum of normal and anomaly likelihoods
            log_likelihood_corrected = jnp.maximum(log_likelihood_with_prior, anomaly_threshold)
            
            # Sum over all points
            return jnp.sum(log_likelihood_corrected)
    else:
        def loglikelihood_fn(theta):
            # Get 2D model prediction
            model_2d = model_func(t_jax, freq_jax, theta, max_peaks, fit_pulses, ref_freq)
            
            # Get sigma (assumes single sigma for all channels for now)
            sigma = theta[sigma_idx]
            
            # Calculate residuals
            residuals = data_2d_jax - model_2d
            
            # If we have per-channel noise, use it for weighting
            if noise_per_channel is not None and len(noise_jax.shape) > 0:
                # Weight by per-channel noise
                weighted_residuals = residuals / noise_jax[:, None]
                log_likelihood = -0.5 * jnp.sum(weighted_residuals**2)
                # Add normalization terms
                log_likelihood -= jnp.sum(jnp.log(noise_jax)) * len(t_jax)
                log_likelihood -= len(data_2d_jax.flatten()) * jnp.log(jnp.sqrt(2 * jnp.pi))
            else:
                # Use single sigma for all points
                log_likelihood = -0.5 * jnp.sum((residuals / sigma) ** 2)
                n_total = data_2d_jax.size
                log_likelihood -= n_total * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
            
            return log_likelihood
    
    # Create prior log-probability function
    import distrax
    bounds = priors.prior_bounds
    
    # Build list of distributions including spectral index
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
    
    # Arrival times - handled specially for sorting
    for i in range(max_peaks):
        dists.append(None)  # Handled in logprior_fn
    
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
            low=bounds.get('baseline', {'min': -1.0, 'max': 1.0})['min'],
            high=bounds.get('baseline', {'min': -1.0, 'max': 1.0})['max']
        ))
    
    # Spectral index - uniform (typical range)
    alpha_bounds = bounds.get('spectral_index', {'min': -3.0, 'max': 1.0})
    dists.append(distrax.Uniform(
        low=alpha_bounds['min'],
        high=alpha_bounds['max']
    ))
    
    # Sigma - uniform
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
    
    # Anomaly probability (if using RFI mitigation) - log-uniform
    if use_rfi_mitigation:
        # log(p) uniform from -10 to -0.1 (p from ~0.00005 to ~0.9)
        log_p_bounds = bounds.get('log_anomaly_prob', {'min': -10.0, 'max': -0.1})
        dists.append(distrax.Uniform(
            low=log_p_bounds['min'],
            high=log_p_bounds['max']
        ))
    
    @jit
    def logprior_fn(theta):
        logp = 0.0
        
        # Amplitudes and Taus - regular priors
        for i in range(2 * max_peaks):
            if dists[i] is not None:
                logp += dists[i].log_prob(theta[i])
        
        # Arrival times - enforce sorting constraint
        u_start = 2 * max_peaks
        u_end = 3 * max_peaks if 'exponential' in model_name else u_start + max_peaks
        u_values = theta[u_start:u_end]
        
        if max_peaks > 1:
            # Check if values are sorted
            is_sorted = jnp.all(u_values[:-1] <= u_values[1:])
            # If not sorted, return -inf (invalid)
            logp = jnp.where(is_sorted, logp, -jnp.inf)
        
        # Check bounds for u values
        u_min = bounds['u']['min']
        u_max = bounds['u']['max']
        in_bounds = jnp.all((u_values >= u_min) & (u_values <= u_max))
        logp = jnp.where(in_bounds, logp, -jnp.inf)
        
        # Uniform prior on sorted values
        logp += -max_peaks * jnp.log(u_max - u_min)
        
        # Continue with remaining parameters
        start_idx = 3 * max_peaks if 'exponential' in model_name else 4 * max_peaks
        for i in range(start_idx, len(dists)):
            if dists[i] is not None:
                logp += dists[i].log_prob(theta[i])
        
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
    
    # Sample initial points from the prior
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
    pbar = tqdm.tqdm(desc="Dead points (2D)", unit=" dead points")
    
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
    
    print("\n" + "="*60)
    print("2D SAMPLING COMPLETED - Starting post-processing")
    print("="*60)
    
    # Clear memory and finalise
    import gc
    gc.collect()
    jax.clear_caches()
    
    # Use chunked finalise
    chunk_size = max(1, len(dead) // 10)
    final_state = finalise_chunked(state, dead, chunk_size=chunk_size)
    
    # Fix NaN birth likelihoods
    logL_birth = np.array(final_state.loglikelihood_birth)
    nan_mask = np.isnan(logL_birth)
    if np.any(nan_mask):
        max_valid = np.nanmax(logL_birth)
        logL_birth[nan_mask] = max_valid
        final_state = final_state._replace(loglikelihood_birth=logL_birth)
    
    return final_state


