"""
Prior transformations for FRB models using JAX.
"""
import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Tuple
import numpy as np


@jit
def uniform_prior(x: float, low: float, high: float) -> float:
    """Transform from unit hypercube to uniform prior."""
    return low + (high - low) * x


@jit
def log_uniform_prior(x: float, low: float, high: float) -> float:
    """Transform from unit hypercube to log-uniform prior."""
    log_low = jnp.log(low)
    log_high = jnp.log(high)
    return jnp.exp(log_low + (log_high - log_low) * x)


def sorted_uniform_prior(x: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
    """
    Transform from unit hypercube to sorted uniform prior.
    This ensures u1 < u2 < ... < un.
    """
    n = len(x)
    if n == 0:
        return jnp.array([])
    elif n == 1:
        return jnp.array([uniform_prior(x[0], low, high)])
    else:
        # Simple approach: transform to uniform then sort
        values = jnp.array([uniform_prior(x[i], low, high) for i in range(n)])
        return jnp.sort(values)


def transform_prior_emg(hypercube: jnp.ndarray, prior_ranges: Dict, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Transform unit hypercube to EMG model parameters.
    
    Args:
        hypercube: Unit hypercube samples
        prior_ranges: Dictionary of prior ranges
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
    
    Returns:
        Transformed parameters
    """
    ndims = 4 * max_peaks + 1 + (1 if fit_pulses else 0)
    theta = jnp.zeros(ndims)
    idx = 0
    
    # Amplitudes
    for i in range(max_peaks):
        theta = theta.at[i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["amplitude"]["min"], prior_ranges["amplitude"]["max"])
        )
    idx += max_peaks
    
    # Tau values
    for i in range(max_peaks):
        theta = theta.at[max_peaks + i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["tau"]["min"], prior_ranges["tau"]["max"])
        )
    idx += max_peaks
    
    # Arrival times - handle sorting for active pulses
    u_hypercube = hypercube[idx:idx + max_peaks]
    idx += max_peaks
    
    if fit_pulses:
        # Get Npulse first to know how many to sort
        Npulse_raw = uniform_prior(hypercube[-1], 1, max_peaks + 1)
        Npulse = jnp.round(Npulse_raw).astype(jnp.int32)
        Npulse = jnp.clip(Npulse, 1, max_peaks)
        
        # Sort first Npulse arrival times
        u_sorted = sorted_uniform_prior(
            u_hypercube[:Npulse], 
            prior_ranges["u"]["min"], 
            prior_ranges["u"]["max"]
        )
        
        # Uniform for remaining
        u_uniform = jnp.array([
            uniform_prior(u_hypercube[i], prior_ranges["u"]["min"], prior_ranges["u"]["max"])
            for i in range(Npulse, max_peaks)
        ])
        
        # Combine
        u_all = jnp.concatenate([u_sorted, u_uniform])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_all[i])
    else:
        # All arrival times sorted
        u_sorted = sorted_uniform_prior(u_hypercube, prior_ranges["u"]["min"], prior_ranges["u"]["max"])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_sorted[i])
    
    # Width parameters
    for i in range(max_peaks):
        theta = theta.at[3 * max_peaks + i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["width"]["min"], prior_ranges["width"]["max"])
        )
    idx += max_peaks
    
    # Sigma
    theta = theta.at[4 * max_peaks].set(
        log_uniform_prior(hypercube[idx], prior_ranges["sigma"]["min"], prior_ranges["sigma"]["max"])
    )
    idx += 1
    
    # Npulse
    if fit_pulses:
        theta = theta.at[-1].set(Npulse_raw)
    
    return theta


def transform_prior_exponential(hypercube: jnp.ndarray, prior_ranges: Dict, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Transform unit hypercube to exponential model parameters.
    
    Args:
        hypercube: Unit hypercube samples
        prior_ranges: Dictionary of prior ranges
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
    
    Returns:
        Transformed parameters
    """
    ndims = 3 * max_peaks + 1 + (1 if fit_pulses else 0)
    theta = jnp.zeros(ndims)
    idx = 0
    
    # Amplitudes
    for i in range(max_peaks):
        theta = theta.at[i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["amplitude"]["min"], prior_ranges["amplitude"]["max"])
        )
    idx += max_peaks
    
    # Tau values (log-uniform for exponential)
    for i in range(max_peaks):
        theta = theta.at[max_peaks + i].set(
            log_uniform_prior(hypercube[idx + i], prior_ranges["tau"]["min"], prior_ranges["tau"]["max"])
        )
    idx += max_peaks
    
    # Arrival times - handle sorting for active pulses
    u_hypercube = hypercube[idx:idx + max_peaks]
    idx += max_peaks
    
    if fit_pulses:
        # Get Npulse first to know how many to sort
        Npulse_raw = uniform_prior(hypercube[-1], 1, max_peaks + 1)
        Npulse = jnp.round(Npulse_raw).astype(jnp.int32)
        Npulse = jnp.clip(Npulse, 1, max_peaks)
        
        # Sort first Npulse arrival times
        u_sorted = sorted_uniform_prior(
            u_hypercube[:Npulse], 
            prior_ranges["u"]["min"], 
            prior_ranges["u"]["max"]
        )
        
        # Uniform for remaining
        u_uniform = jnp.array([
            uniform_prior(u_hypercube[i], prior_ranges["u"]["min"], prior_ranges["u"]["max"])
            for i in range(Npulse, max_peaks)
        ])
        
        # Combine
        u_all = jnp.concatenate([u_sorted, u_uniform])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_all[i])
    else:
        # All arrival times sorted
        u_sorted = sorted_uniform_prior(u_hypercube, prior_ranges["u"]["min"], prior_ranges["u"]["max"])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_sorted[i])
    
    # Sigma
    theta = theta.at[3 * max_peaks].set(
        log_uniform_prior(hypercube[idx], prior_ranges["sigma"]["min"], prior_ranges["sigma"]["max"])
    )
    idx += 1
    
    # Npulse
    if fit_pulses:
        theta = theta.at[-1].set(Npulse_raw)
    
    return theta


def transform_prior_emg_with_baseline(hypercube: jnp.ndarray, prior_ranges: Dict, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Transform unit hypercube to EMG model with baseline parameters.
    """
    ndims = 4 * max_peaks + 2 + (1 if fit_pulses else 0)
    theta = jnp.zeros(ndims)
    idx = 0
    
    # Amplitudes
    for i in range(max_peaks):
        theta = theta.at[i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["amplitude"]["min"], prior_ranges["amplitude"]["max"])
        )
    idx += max_peaks
    
    # Tau values
    for i in range(max_peaks):
        theta = theta.at[max_peaks + i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["tau"]["min"], prior_ranges["tau"]["max"])
        )
    idx += max_peaks
    
    # Arrival times - handle sorting for active pulses
    u_hypercube = hypercube[idx:idx + max_peaks]
    idx += max_peaks
    
    if fit_pulses:
        # Get Npulse first to know how many to sort
        Npulse_raw = uniform_prior(hypercube[-1], 1, max_peaks + 1)
        Npulse = jnp.round(Npulse_raw).astype(jnp.int32)
        Npulse = jnp.clip(Npulse, 1, max_peaks)
        
        # Sort first Npulse arrival times
        u_sorted = sorted_uniform_prior(
            u_hypercube[:Npulse], 
            prior_ranges["u"]["min"], 
            prior_ranges["u"]["max"]
        )
        
        # Uniform for remaining
        u_uniform = jnp.array([
            uniform_prior(u_hypercube[i], prior_ranges["u"]["min"], prior_ranges["u"]["max"])
            for i in range(Npulse, max_peaks)
        ])
        
        # Combine
        u_all = jnp.concatenate([u_sorted, u_uniform])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_all[i])
    else:
        # All arrival times sorted
        u_sorted = sorted_uniform_prior(u_hypercube, prior_ranges["u"]["min"], prior_ranges["u"]["max"])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_sorted[i])
    
    # Width parameters
    for i in range(max_peaks):
        theta = theta.at[3 * max_peaks + i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["width"]["min"], prior_ranges["width"]["max"])
        )
    idx += max_peaks
    
    # Baseline offset
    theta = theta.at[4 * max_peaks].set(
        uniform_prior(hypercube[idx], prior_ranges["baseline_offset"]["min"], prior_ranges["baseline_offset"]["max"])
    )
    idx += 1
    
    # Sigma
    theta = theta.at[4 * max_peaks + 1].set(
        log_uniform_prior(hypercube[idx], prior_ranges["sigma"]["min"], prior_ranges["sigma"]["max"])
    )
    idx += 1
    
    # Npulse
    if fit_pulses:
        theta = theta.at[-1].set(Npulse_raw)
    
    return theta


def transform_prior_exponential_with_baseline(hypercube: jnp.ndarray, prior_ranges: Dict, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Transform unit hypercube to exponential model with baseline parameters.
    """
    ndims = 3 * max_peaks + 2 + (1 if fit_pulses else 0)
    theta = jnp.zeros(ndims)
    idx = 0
    
    # Amplitudes
    for i in range(max_peaks):
        theta = theta.at[i].set(
            uniform_prior(hypercube[idx + i], prior_ranges["amplitude"]["min"], prior_ranges["amplitude"]["max"])
        )
    idx += max_peaks
    
    # Tau values (log-uniform for exponential)
    for i in range(max_peaks):
        theta = theta.at[max_peaks + i].set(
            log_uniform_prior(hypercube[idx + i], prior_ranges["tau"]["min"], prior_ranges["tau"]["max"])
        )
    idx += max_peaks
    
    # Arrival times - handle sorting for active pulses
    u_hypercube = hypercube[idx:idx + max_peaks]
    idx += max_peaks
    
    if fit_pulses:
        # Get Npulse first to know how many to sort
        Npulse_raw = uniform_prior(hypercube[-1], 1, max_peaks + 1)
        Npulse = jnp.round(Npulse_raw).astype(jnp.int32)
        Npulse = jnp.clip(Npulse, 1, max_peaks)
        
        # Sort first Npulse arrival times
        u_sorted = sorted_uniform_prior(
            u_hypercube[:Npulse], 
            prior_ranges["u"]["min"], 
            prior_ranges["u"]["max"]
        )
        
        # Uniform for remaining
        u_uniform = jnp.array([
            uniform_prior(u_hypercube[i], prior_ranges["u"]["min"], prior_ranges["u"]["max"])
            for i in range(Npulse, max_peaks)
        ])
        
        # Combine
        u_all = jnp.concatenate([u_sorted, u_uniform])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_all[i])
    else:
        # All arrival times sorted
        u_sorted = sorted_uniform_prior(u_hypercube, prior_ranges["u"]["min"], prior_ranges["u"]["max"])
        for i in range(max_peaks):
            theta = theta.at[2 * max_peaks + i].set(u_sorted[i])
    
    # Baseline offset
    theta = theta.at[3 * max_peaks].set(
        uniform_prior(hypercube[idx], prior_ranges["baseline_offset"]["min"], prior_ranges["baseline_offset"]["max"])
    )
    idx += 1
    
    # Sigma
    theta = theta.at[3 * max_peaks + 1].set(
        log_uniform_prior(hypercube[idx], prior_ranges["sigma"]["min"], prior_ranges["sigma"]["max"])
    )
    idx += 1
    
    # Npulse
    if fit_pulses:
        theta = theta.at[-1].set(Npulse_raw)
    
    return theta


def get_prior_transform(model_name: str, prior_ranges: Dict, max_peaks: int, fit_pulses: bool) -> callable:
    """
    Get the prior transformation function for a given model.
    
    Args:
        model_name: Name of the model
        prior_ranges: Dictionary of prior ranges
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit number of pulses
    
    Returns:
        Prior transformation function
    """
    if model_name == "emg":
        return lambda x: transform_prior_emg(x, prior_ranges, max_peaks, fit_pulses)
    elif model_name == "exponential":
        return lambda x: transform_prior_exponential(x, prior_ranges, max_peaks, fit_pulses)
    elif model_name == "emg_with_baseline":
        return lambda x: transform_prior_emg_with_baseline(x, prior_ranges, max_peaks, fit_pulses)
    elif model_name == "exponential_with_baseline":
        return lambda x: transform_prior_exponential_with_baseline(x, prior_ranges, max_peaks, fit_pulses)
    else:
        raise ValueError(f"Model {model_name} not recognized")