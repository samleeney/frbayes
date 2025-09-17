"""
3D Basis Function Sampling Extension for FRBayes.
This module extends sampling.py with 3D basis function capabilities.
"""

import jax
import jax.numpy as jnp
from jax import jit
import blackjax
import numpy as np
from typing import Dict, Optional
import tqdm
from .models import get_model_function
from .priors import FRBPriors3D
from .sampling import finalise_chunked


def run_nested_sampling_3d_basis(
    model_name: str,
    data_3d: np.ndarray,
    t: np.ndarray,
    freq: np.ndarray,
    n_basis: int = 5,
    noise_per_channel: Optional[np.ndarray] = None,
    prior_bounds: Optional[Dict] = None,
    max_peaks: int = 2,
    fit_pulses: bool = False,
    use_rfi_mitigation: bool = False,
    num_live_points: int = 1000,
    num_delete: int = 50,
    num_inner_steps: int = 20,
    log_tolerance: float = -3.0,
    seed: int = 0,
    ref_freq: float = 1400.0
):
    """
    Run nested sampling for 3D basis function models.

    Args:
        model_name: Name of the 3D model ("emg_3d_basis" or "exponential_3d_basis")
        data_3d: 3D waterfall data (freq, time)
        t: Time axis
        freq: Frequency axis (MHz)
        n_basis: Number of basis functions for spectral modeling
        noise_per_channel: Per-channel noise estimates (if None, uses single sigma)
        prior_bounds: Prior bounds for parameters (uses defaults if None)
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
        use_rfi_mitigation: Whether to use Bayesian RFI mitigation
        num_live_points: Number of live points
        num_delete: Number of points to delete per iteration
        num_inner_steps: Number of MCMC steps between replacements
        log_tolerance: Termination criterion (log(Z_live/Z))
        seed: Random seed
        ref_freq: Reference frequency in MHz for normalization

    Returns:
        Dictionary containing sampling results
    """
    # Ensure we have a 3D basis model
    if "3d_basis" not in model_name:
        raise ValueError(f"Model {model_name} is not a 3D basis model. Use 'emg_3d_basis' or 'exponential_3d_basis'")

    # Convert data to JAX arrays
    data_3d_jax = jnp.array(data_3d)
    t_jax = jnp.array(t)
    freq_jax = jnp.array(freq)

    if noise_per_channel is not None:
        noise_jax = jnp.array(noise_per_channel)
    else:
        # Estimate single noise level from the data
        noise_jax = jnp.std(data_3d_jax)

    # Initialize prior system for 3D models
    priors = FRBPriors3D(model_name, max_peaks, n_basis, fit_pulses, prior_bounds, use_rfi_mitigation)
    ndims = priors.ndims

    # Get model function
    model_func = get_model_function(model_name)

    # Calculate parameter indices for 3D model
    if model_name == "emg_3d_basis":
        baseline_idx = 4 * max_peaks + n_basis  # After A, tau, u, w, c0...c{K-1}
        sigma_idx = baseline_idx + 1  # After baseline
    elif model_name == "exponential_3d_basis":
        baseline_idx = 3 * max_peaks + n_basis  # After A, tau, u, c0...c{K-1}
        sigma_idx = baseline_idx + 1  # After baseline
    else:
        raise ValueError(f"Unknown 3D model: {model_name}")

    # Calculate log_p index if using RFI mitigation
    if use_rfi_mitigation:
        log_p_idx = sigma_idx + 1 + (1 if fit_pulses else 0)

    # Create log-likelihood function
    def loglikelihood_fn(theta):
        # Get 3D model prediction
        model_pred = model_func(t_jax, freq_jax, theta, max_peaks, n_basis, fit_pulses, ref_freq)

        # Get sigma
        sigma = theta[sigma_idx]

        # Calculate residuals
        residuals = data_3d_jax - model_pred

        if use_rfi_mitigation:
            # Get anomaly probability
            log_p = theta[log_p_idx]
            p = jnp.exp(log_p)  # Anomaly probability

            # Calculate delta (maximum expected deviation)
            # Use 99.9th percentile of absolute data values
            delta = jnp.percentile(jnp.abs(data_3d_jax), 99.9)
            delta = jnp.maximum(delta, 1.0)  # Ensure delta is at least 1.0

            if noise_per_channel is not None:
                # Per-channel noise with RFI mitigation
                normalized_residuals = residuals / noise_jax[:, None]
                # Normal likelihood for each pixel
                logL_normal = -0.5 * normalized_residuals ** 2 - jnp.log(noise_jax[:, None] * jnp.sqrt(2 * jnp.pi))
            else:
                # Single sigma with RFI mitigation
                logL_normal = -0.5 * (residuals / sigma) ** 2 - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

            # Add prior probability of being normal data (not anomaly)
            logL_with_prior = logL_normal + jnp.log(1 - p)

            # Anomaly threshold (uniform likelihood over [-delta, delta])
            logL_anomaly = log_p - jnp.log(2 * delta)  # Uniform over range

            # Piecewise likelihood: max(normal, anomaly) for each pixel
            logL_corrected = jnp.maximum(logL_with_prior, logL_anomaly)

            # Sum over all pixels
            log_likelihood = jnp.sum(logL_corrected)
        else:
            # Standard likelihood without RFI mitigation
            if noise_per_channel is not None:
                # Use per-channel noise weighting
                normalized_residuals = residuals / noise_jax[:, None]
                n_total = residuals.size
                log_likelihood = -0.5 * jnp.sum(normalized_residuals ** 2) - n_total * jnp.log(jnp.sqrt(2 * jnp.pi))
                # Add log-determinant term for per-channel noise
                log_likelihood -= jnp.sum(jnp.log(noise_jax)) * len(t_jax)
            else:
                # Use single sigma for all data
                n_total = residuals.size
                log_likelihood = -0.5 * jnp.sum((residuals / sigma) ** 2) - n_total * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

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

    # Arrival times - handled specially below with sorting
    for i in range(max_peaks):
        dists.append(None)  # Will be handled in logprior_fn

    # Widths (for EMG) - uniform
    if 'emg' in model_name:
        for i in range(max_peaks):
            dists.append(distrax.Uniform(
                low=bounds['width']['min'],
                high=bounds['width']['max']
            ))

    # Basis function coefficients - uniform with regularization
    for k in range(n_basis):
        if k == 0:
            # Zeroth order - wider range
            coeff_min = bounds['spectral_coeffs']['min']
            coeff_max = bounds['spectral_coeffs']['max']
        else:
            # Higher order - progressively narrower range
            scale = 1.0 / (k + 1)
            coeff_min = bounds['spectral_coeffs']['min'] * scale
            coeff_max = bounds['spectral_coeffs']['max'] * scale

        dists.append(distrax.Uniform(low=coeff_min, high=coeff_max))

    # Sigma - log-uniform (use uniform in log space)
    # distrax doesn't have LogUniform, so we use None and handle in logprior_fn
    dists.append(None)

    # Npulse (if fitted) - uniform
    if fit_pulses:
        dists.append(distrax.Uniform(low=1.0, high=float(max_peaks)))

    # Create log-prior function
    def logprior_fn(theta):
        logp = 0.0

        # Amplitudes and Taus - regular priors
        for i in range(2 * max_peaks):
            if dists[i] is not None:
                logp += dists[i].log_prob(theta[i])

        # Arrival times - enforce sorting constraint
        u_start = 2 * max_peaks
        u_end = u_start + max_peaks
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

        # Continue with remaining parameters (widths if EMG)
        if 'emg' in model_name:
            # Add width priors
            for i in range(max_peaks):
                width_idx = 3 * max_peaks + i
                if width_idx < len(dists) and dists[width_idx] is not None:
                    logp += dists[width_idx].log_prob(theta[width_idx])

        # Basis function coefficients
        coeff_start = 3 * max_peaks if 'exponential' in model_name else 4 * max_peaks
        for k in range(n_basis):
            coeff_idx = coeff_start + k
            if coeff_idx < len(dists) and dists[coeff_idx] is not None:
                logp += dists[coeff_idx].log_prob(theta[coeff_idx])

        # Handle sigma (log-uniform)
        sigma_idx = coeff_start + n_basis
        sigma = theta[sigma_idx]
        log_sigma_min = bounds['log_sigma']['min']
        log_sigma_max = bounds['log_sigma']['max']
        # Log-uniform prior: p(sigma) = 1/(sigma * log(max/min))
        sigma_in_bounds = (sigma >= jnp.exp(log_sigma_min)) & (sigma <= jnp.exp(log_sigma_max))
        logp = jnp.where(sigma_in_bounds,
                        logp - jnp.log(sigma) - jnp.log(log_sigma_max - log_sigma_min),
                        -jnp.inf)

        # Handle Npulse if fitted
        if fit_pulses:
            npulse_idx = sigma_idx + 1
            if npulse_idx < len(dists) and dists[npulse_idx] is not None:
                logp += dists[npulse_idx].log_prob(theta[npulse_idx])

        # Handle log_p for RFI mitigation if enabled
        if use_rfi_mitigation:
            log_p_idx = sigma_idx + 1 + (1 if fit_pulses else 0)
            log_p = theta[log_p_idx]
            log_p_bounds = bounds.get('log_anomaly_prob', {'min': -10.0, 'max': -0.1})
            # Uniform prior on log_p
            log_p_in_bounds = (log_p >= log_p_bounds['min']) & (log_p <= log_p_bounds['max'])
            logp = jnp.where(log_p_in_bounds,
                            logp - jnp.log(log_p_bounds['max'] - log_p_bounds['min']),
                            -jnp.inf)

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
    pbar = tqdm.tqdm(desc="Dead points (3D basis)", unit=" dead points")

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
    print("3D BASIS SAMPLING COMPLETED - Starting post-processing")
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