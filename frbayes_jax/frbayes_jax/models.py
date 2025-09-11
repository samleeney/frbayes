"""
JAX-based FRB pulse models.
All models are JIT-compilable and differentiable.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import erfc
from typing import Dict, Tuple, Callable, Optional
import numpy as np


@jit
def emg_pulse(t: jnp.ndarray, A: float, tau: float, u: float, w: float) -> jnp.ndarray:
    """
    Exponentially Modified Gaussian pulse model for a single pulse.
    
    Args:
        t: Time array
        A: Amplitude
        tau: Exponential decay time constant
        u: Peak arrival time
        w: Gaussian width parameter
    
    Returns:
        EMG pulse evaluated at time t
    """
    # Proper EMG implementation using JAX-compatible error function
    # Based on the original formula from archive/frbayes_cpu/frbayes/models.py
    from jax.scipy.special import erfc
    
    # Direct calculation without safeguards
    exp_arg = ((u - t) / tau) + ((w ** 2) / (2 * tau ** 2))
    
    # Calculate erfc argument
    erfc_arg = (((u - t) * tau) + w ** 2) / (w * tau * jnp.sqrt(2))
    
    # EMG formula: combination of exponential decay and Gaussian convolution
    emg = (A / (2 * tau)) * jnp.exp(exp_arg) * erfc(erfc_arg)
    
    return emg


@jit
def exponential_pulse(t: jnp.ndarray, A: float, tau: float, u: float) -> jnp.ndarray:
    """
    Simple exponential decay pulse model for a single pulse.
    
    Args:
        t: Time array
        A: Amplitude
        tau: Exponential decay time constant
        u: Peak arrival time
    
    Returns:
        Exponential pulse evaluated at time t
    """
    # Pure exponential with sharp cutoff at arrival time
    # Using jnp.where for a true step function
    return jnp.where(t >= u, A * jnp.exp(-(t - u) / tau), 0.0)


def emg_model(t: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Multi-pulse EMG model.
    
    Args:
        t: Time array
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, w1,...,wn, sigma, (Npulse)]
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
    
    Returns:
        Model prediction at time t
    """
    # Extract parameters using static indices
    A = jax.lax.dynamic_slice(theta, (0,), (max_peaks,))
    tau = jax.lax.dynamic_slice(theta, (max_peaks,), (max_peaks,))
    u = jax.lax.dynamic_slice(theta, (2*max_peaks,), (max_peaks,))
    w = jax.lax.dynamic_slice(theta, (3*max_peaks,), (max_peaks,))
    
    if fit_pulses:
        Npulse = jnp.round(theta[-1])
        Npulse = jnp.clip(Npulse, 1, max_peaks)
    else:
        Npulse = max_peaks
    
    # Sum contributions from all pulses
    model = jnp.zeros_like(t)
    for i in range(max_peaks):
        # Only include pulse if i < Npulse
        pulse_active = i < Npulse
        contribution = emg_pulse(t, A[i], tau[i], u[i], w[i])
        model += jnp.where(pulse_active, contribution, 0.0)
    
    return model


def exponential_model(t: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Multi-pulse exponential model.
    
    Args:
        t: Time array
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, sigma, (Npulse)]
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
    
    Returns:
        Model prediction at time t
    """
    # Extract parameters using static indices
    A = jax.lax.dynamic_slice(theta, (0,), (max_peaks,))
    tau = jax.lax.dynamic_slice(theta, (max_peaks,), (max_peaks,))
    u = jax.lax.dynamic_slice(theta, (2*max_peaks,), (max_peaks,))
    
    if fit_pulses:
        Npulse = jnp.round(theta[-1])
        Npulse = jnp.clip(Npulse, 1, max_peaks)
    else:
        Npulse = max_peaks
    
    # Sum contributions from all pulses
    model = jnp.zeros_like(t)
    for i in range(max_peaks):
        # Only include pulse if i < Npulse
        pulse_active = i < Npulse
        contribution = exponential_pulse(t, A[i], tau[i], u[i])
        model += jnp.where(pulse_active, contribution, 0.0)
    
    return model


def emg_model_with_baseline(t: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Multi-pulse EMG model with baseline offset.
    
    Args:
        t: Time array
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, w1,...,wn, B_offset, sigma, (Npulse)]
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
    
    Returns:
        Model prediction at time t
    """
    # Extract parameters using static indices
    A = jax.lax.dynamic_slice(theta, (0,), (max_peaks,))
    tau = jax.lax.dynamic_slice(theta, (max_peaks,), (max_peaks,))
    u = jax.lax.dynamic_slice(theta, (2*max_peaks,), (max_peaks,))
    w = jax.lax.dynamic_slice(theta, (3*max_peaks,), (max_peaks,))
    baseline_offset = theta[4*max_peaks]
    
    if fit_pulses:
        Npulse = jnp.round(theta[-1])
        Npulse = jnp.clip(Npulse, 1, max_peaks)
    else:
        Npulse = max_peaks
    
    # Sum contributions from all pulses
    model = jnp.zeros_like(t)
    for i in range(max_peaks):
        # Only include pulse if i < Npulse
        pulse_active = i < Npulse
        contribution = emg_pulse(t, A[i], tau[i], u[i], w[i])
        model += jnp.where(pulse_active, contribution, 0.0)
    
    return model + baseline_offset


def exponential_model_with_baseline(t: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Multi-pulse exponential model with baseline offset.
    
    Args:
        t: Time array
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, B_offset, sigma, (Npulse)]
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
    
    Returns:
        Model prediction at time t
    """
    # Extract parameters using static indices
    A = jax.lax.dynamic_slice(theta, (0,), (max_peaks,))
    tau = jax.lax.dynamic_slice(theta, (max_peaks,), (max_peaks,))
    u = jax.lax.dynamic_slice(theta, (2*max_peaks,), (max_peaks,))
    baseline_offset = theta[3*max_peaks]
    
    if fit_pulses:
        Npulse = jnp.round(theta[-1])
        Npulse = jnp.clip(Npulse, 1, max_peaks)
    else:
        Npulse = max_peaks
    
    # Sum contributions from all pulses
    model = jnp.zeros_like(t)
    for i in range(max_peaks):
        # Only include pulse if i < Npulse
        pulse_active = i < Npulse
        contribution = exponential_pulse(t, A[i], tau[i], u[i])
        model += jnp.where(pulse_active, contribution, 0.0)
    
    return model + baseline_offset


def periodic_exponential_model(t: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Periodic exponential model with regularly spaced pulses.
    
    Args:
        t: Time array
        theta: Parameter array [A1,...,An, tau1,...,taun, u0, period, sigma, (Npulse)]
               where u0 is the first pulse location and period is the spacing
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
    
    Returns:
        Model prediction at time t
    """
    # Extract parameters using static indices
    A = jax.lax.dynamic_slice(theta, (0,), (max_peaks,))
    tau = jax.lax.dynamic_slice(theta, (max_peaks,), (max_peaks,))
    u0 = theta[2*max_peaks]  # First pulse location
    period = theta[2*max_peaks + 1]  # Period between pulses
    
    if fit_pulses:
        Npulse = jnp.round(theta[-1])
        Npulse = jnp.clip(Npulse, 1, max_peaks)
    else:
        Npulse = max_peaks
    
    # Sum contributions from all pulses
    model = jnp.zeros_like(t)
    for i in range(max_peaks):
        # Only include pulse if i < Npulse
        pulse_active = i < Npulse
        # Calculate pulse location: u_i = u0 + i * period
        u_i = u0 + i * period
        contribution = exponential_pulse(t, A[i], tau[i], u_i)
        model += jnp.where(pulse_active, contribution, 0.0)
    
    return model


def periodic_exponential_model_with_baseline(t: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool) -> jnp.ndarray:
    """
    Periodic exponential model with regularly spaced pulses and baseline offset.
    
    Args:
        t: Time array
        theta: Parameter array [A1,...,An, tau1,...,taun, u0, period, B_offset, sigma, (Npulse)]
               where u0 is the first pulse location and period is the spacing
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
    
    Returns:
        Model prediction at time t
    """
    # Extract parameters using static indices
    A = jax.lax.dynamic_slice(theta, (0,), (max_peaks,))
    tau = jax.lax.dynamic_slice(theta, (max_peaks,), (max_peaks,))
    u0 = theta[2*max_peaks]  # First pulse location
    period = theta[2*max_peaks + 1]  # Period between pulses
    baseline_offset = theta[2*max_peaks + 2]  # Baseline offset
    
    if fit_pulses:
        Npulse = jnp.round(theta[-1])
        Npulse = jnp.clip(Npulse, 1, max_peaks)
    else:
        Npulse = max_peaks
    
    # Sum contributions from all pulses
    model = jnp.zeros_like(t)
    for i in range(max_peaks):
        # Only include pulse if i < Npulse
        pulse_active = i < Npulse
        # Calculate pulse location: u_i = u0 + i * period
        u_i = u0 + i * period
        contribution = exponential_pulse(t, A[i], tau[i], u_i)
        model += jnp.where(pulse_active, contribution, 0.0)
    
    return model + baseline_offset


def get_model_function(model_name: str) -> Callable:
    """
    Get the model function for a given model name.
    
    Args:
        model_name: Name of the model
    
    Returns:
        JIT-compiled model function
    """
    models = {
        "emg": emg_model,
        "exponential": exponential_model,
        "emg_with_baseline": emg_model_with_baseline,
        "exponential_with_baseline": exponential_model_with_baseline,
        "periodic_exponential": periodic_exponential_model,
        "periodic_exponential_with_baseline": periodic_exponential_model_with_baseline
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not recognized. Available: {list(models.keys())}")
    
    return models[model_name]


def get_num_params(model_name: str, max_peaks: int, fit_pulses: bool) -> int:
    """
    Get the number of parameters for a given model.
    
    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit the number of pulses
    
    Returns:
        Number of parameters
    """
    if model_name == "emg":
        # A, tau, u, w for each peak + sigma + (optionally) Npulse
        ndims = 4 * max_peaks + 1
    elif model_name == "exponential":
        # A, tau, u for each peak + sigma + (optionally) Npulse
        ndims = 3 * max_peaks + 1
    elif model_name == "emg_with_baseline":
        # A, tau, u, w for each peak + baseline + sigma + (optionally) Npulse
        ndims = 4 * max_peaks + 2
    elif model_name == "exponential_with_baseline":
        # A, tau, u for each peak + baseline + sigma + (optionally) Npulse
        ndims = 3 * max_peaks + 2
    elif model_name == "periodic_exponential":
        # A, tau for each peak + u0 + period + sigma + (optionally) Npulse
        ndims = 2 * max_peaks + 3
    elif model_name == "periodic_exponential_with_baseline":
        # A, tau for each peak + u0 + period + baseline + sigma + (optionally) Npulse
        ndims = 2 * max_peaks + 4
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    if fit_pulses:
        ndims += 1
    
    return ndims


def get_param_names(model_name: str, max_peaks: int, fit_pulses: bool) -> list:
    """
    Get parameter names for a given model.
    
    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit the number of pulses
    
    Returns:
        List of parameter names in LaTeX format
    """
    names = []
    
    # Amplitude parameters
    for i in range(max_peaks):
        names.append(rf"$A_{{{i+1}}}$")
    
    # Tau parameters
    for i in range(max_peaks):
        names.append(rf"$\tau_{{{i+1}}}$")
    
    # Model-specific location parameters
    if "periodic" in model_name:
        # For periodic models: u0 and period
        names.append(r"$u_0$")
        names.append(r"$T$")  # Period
    else:
        # For non-periodic models: individual arrival times
        for i in range(max_peaks):
            names.append(rf"$u_{{{i+1}}}$")
    
    # Width parameters (for EMG models)
    if "emg" in model_name and "periodic" not in model_name:
        for i in range(max_peaks):
            names.append(rf"$w_{{{i+1}}}$")
    
    # Baseline offset (for baseline models)
    if "baseline" in model_name:
        names.append(r"$B_{\text{offset}}$")
    
    # Sigma (noise)
    names.append(r"$\sigma$")
    
    # Npulse (if fitted)
    if fit_pulses:
        names.append(r"$N_{\text{pulse}}$")
    
    return names


def get_sigma_index(model_name: str, max_peaks: int, fit_pulses: bool) -> int:
    """
    Get the index of sigma parameter in theta.
    
    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit the number of pulses
    
    Returns:
        Index of sigma parameter
    """
    if model_name == "emg":
        return 4 * max_peaks
    elif model_name == "exponential":
        return 3 * max_peaks
    elif model_name == "emg_with_baseline":
        return 4 * max_peaks + 1
    elif model_name == "exponential_with_baseline":
        return 3 * max_peaks + 1
    elif model_name == "periodic_exponential":
        return 2 * max_peaks + 2
    elif model_name == "periodic_exponential_with_baseline":
        return 2 * max_peaks + 3
    else:
        raise ValueError(f"Model {model_name} not recognized")


def get_npulse_index(model_name: str, max_peaks: int, fit_pulses: bool) -> Optional[int]:
    """
    Get the index of Npulse parameter in theta.
    
    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit the number of pulses
    
    Returns:
        Index of Npulse parameter or None if not fitted
    """
    if not fit_pulses:
        return None
    
    ndims = get_num_params(model_name, max_peaks, fit_pulses)
    return ndims - 1