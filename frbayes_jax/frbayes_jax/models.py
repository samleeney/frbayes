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


def emg_model_2d(t: jnp.ndarray, freq: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool, ref_freq: float = 1400.0) -> jnp.ndarray:
    """
    2D EMG model with spectral index for dedispersed data.
    
    Args:
        t: Time array
        freq: Frequency array (MHz)
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, w1,...,wn, alpha, sigma, (Npulse)]
               where alpha is the spectral index
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
        ref_freq: Reference frequency in MHz
    
    Returns:
        2D model prediction (freq, time)
    """
    # Get spectral index location
    alpha_idx = 4 * max_peaks
    alpha = theta[alpha_idx]
    
    # Create theta without spectral index for base model
    theta_base = jnp.concatenate([
        theta[:alpha_idx],  # All params before alpha
        theta[alpha_idx+1:]  # sigma and potentially Npulse
    ])
    
    # Get base temporal model (1D)
    base_model = emg_model(t, theta_base, max_peaks, fit_pulses)
    
    # Apply frequency scaling using vmap
    def scale_by_freq(f):
        return base_model * (f / ref_freq) ** alpha
    
    # Vectorize over frequency
    model_2d = vmap(scale_by_freq)(freq)
    
    return model_2d


def exponential_model_2d(t: jnp.ndarray, freq: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool, ref_freq: float = 1400.0) -> jnp.ndarray:
    """
    2D exponential model with spectral index for dedispersed data.
    
    Args:
        t: Time array
        freq: Frequency array (MHz)
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, alpha, sigma, (Npulse)]
               where alpha is the spectral index
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
        ref_freq: Reference frequency in MHz
    
    Returns:
        2D model prediction (freq, time)
    """
    # Get spectral index location
    alpha_idx = 3 * max_peaks
    alpha = theta[alpha_idx]
    
    # Create theta without spectral index for base model
    theta_base = jnp.concatenate([
        theta[:alpha_idx],  # All params before alpha
        theta[alpha_idx+1:]  # sigma and potentially Npulse
    ])
    
    # Get base temporal model (1D)
    base_model = exponential_model(t, theta_base, max_peaks, fit_pulses)
    
    # Apply frequency scaling using vmap
    def scale_by_freq(f):
        return base_model * (f / ref_freq) ** alpha
    
    # Vectorize over frequency
    model_2d = vmap(scale_by_freq)(freq)
    
    return model_2d


def emg_model_2d_with_baseline(t: jnp.ndarray, freq: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool, ref_freq: float = 1400.0) -> jnp.ndarray:
    """
    2D EMG model with spectral index and baseline for dedispersed data.
    
    Args:
        t: Time array
        freq: Frequency array (MHz)
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, w1,...,wn, B_offset, alpha, sigma, (Npulse)]
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
        ref_freq: Reference frequency in MHz
    
    Returns:
        2D model prediction (freq, time)
    """
    # Get spectral index location (after baseline)
    alpha_idx = 4 * max_peaks + 1
    alpha = theta[alpha_idx]
    
    # Create theta without spectral index for base model
    theta_base = jnp.concatenate([
        theta[:alpha_idx],  # All params before alpha including baseline
        theta[alpha_idx+1:]  # sigma and potentially Npulse
    ])
    
    # Get base temporal model with baseline (1D)
    base_model = emg_model_with_baseline(t, theta_base, max_peaks, fit_pulses)
    
    # Extract baseline separately to avoid scaling it
    baseline = theta[4 * max_peaks]
    base_model_no_baseline = base_model - baseline
    
    # Apply frequency scaling to pulse component only
    def scale_by_freq(f):
        return base_model_no_baseline * (f / ref_freq) ** alpha + baseline
    
    # Vectorize over frequency
    model_2d = vmap(scale_by_freq)(freq)
    
    return model_2d


def exponential_model_2d_with_baseline(t: jnp.ndarray, freq: jnp.ndarray, theta: jnp.ndarray, max_peaks: int, fit_pulses: bool, ref_freq: float = 1400.0) -> jnp.ndarray:
    """
    2D exponential model with spectral index and baseline for dedispersed data.
    
    Args:
        t: Time array
        freq: Frequency array (MHz)
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, B_offset, alpha, sigma, (Npulse)]
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is included in theta
        ref_freq: Reference frequency in MHz
    
    Returns:
        2D model prediction (freq, time)
    """
    # Get spectral index location (after baseline)
    alpha_idx = 3 * max_peaks + 1
    alpha = theta[alpha_idx]
    
    # Create theta without spectral index for base model
    theta_base = jnp.concatenate([
        theta[:alpha_idx],  # All params before alpha including baseline
        theta[alpha_idx+1:]  # sigma and potentially Npulse
    ])
    
    # Get base temporal model with baseline (1D)
    base_model = exponential_model_with_baseline(t, theta_base, max_peaks, fit_pulses)
    
    # Extract baseline separately to avoid scaling it
    baseline = theta[3 * max_peaks]
    base_model_no_baseline = base_model - baseline
    
    # Apply frequency scaling to pulse component only
    def scale_by_freq(f):
        return base_model_no_baseline * (f / ref_freq) ** alpha + baseline
    
    # Vectorize over frequency
    model_2d = vmap(scale_by_freq)(freq)
    
    return model_2d


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
        "periodic_exponential_with_baseline": periodic_exponential_model_with_baseline,
        # 2D models
        "emg_2d": emg_model_2d,
        "exponential_2d": exponential_model_2d,
        "emg_2d_with_baseline": emg_model_2d_with_baseline,
        "exponential_2d_with_baseline": exponential_model_2d_with_baseline,
        # 3D basis function models
        "emg_3d_basis": emg_model_3d_basis,
        "exponential_3d_basis": exponential_model_3d_basis
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not recognized. Available: {list(models.keys())}")

    return models[model_name]


def get_num_params(model_name: str, max_peaks: int, fit_pulses: bool, n_basis: int = 5) -> int:
    """
    Get the number of parameters for a given model.

    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit the number of pulses
        n_basis: Number of basis functions for 3D models (default: 5)

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
    # 2D models with spectral index
    elif model_name == "emg_2d":
        # A, tau, u, w for each peak + alpha + sigma + (optionally) Npulse
        ndims = 4 * max_peaks + 2
    elif model_name == "exponential_2d":
        # A, tau, u for each peak + alpha + sigma + (optionally) Npulse
        ndims = 3 * max_peaks + 2
    elif model_name == "emg_2d_with_baseline":
        # A, tau, u, w for each peak + baseline + alpha + sigma + (optionally) Npulse
        ndims = 4 * max_peaks + 3
    elif model_name == "exponential_2d_with_baseline":
        # A, tau, u for each peak + baseline + alpha + sigma + (optionally) Npulse
        ndims = 3 * max_peaks + 3
    # 3D basis function models
    elif model_name == "emg_3d_basis":
        # A, tau, u, w for each peak + c0...c{K-1} + sigma + (optionally) Npulse
        ndims = 4 * max_peaks + n_basis + 1
    elif model_name == "exponential_3d_basis":
        # A, tau, u for each peak + c0...c{K-1} + sigma + (optionally) Npulse
        ndims = 3 * max_peaks + n_basis + 1
    else:
        raise ValueError(f"Model {model_name} not recognized")

    if fit_pulses:
        ndims += 1

    return ndims


def get_param_names(model_name: str, max_peaks: int, fit_pulses: bool, n_basis: int = 5) -> list:
    """
    Get parameter names for a given model.

    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks
        fit_pulses: Whether to fit the number of pulses
        n_basis: Number of basis functions for 3D models

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
    if "baseline" in model_name and "2d" not in model_name:
        names.append(r"$B_{\text{offset}}$")
    elif "baseline" in model_name and "2d" in model_name:
        # For 2D models, baseline comes before spectral index
        names.append(r"$B_{\text{offset}}$")

    # Spectral index (for 2D models)
    if "2d" in model_name and "3d" not in model_name:
        names.append(r"$\alpha$")

    # Basis function coefficients (for 3D models)
    if "3d_basis" in model_name:
        for k in range(n_basis):
            names.append(rf"$c_{{{k}}}$")

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
    # 2D models
    elif model_name == "emg_2d":
        return 4 * max_peaks + 1  # After alpha
    elif model_name == "exponential_2d":
        return 3 * max_peaks + 1  # After alpha
    elif model_name == "emg_2d_with_baseline":
        return 4 * max_peaks + 2  # After baseline and alpha
    elif model_name == "exponential_2d_with_baseline":
        return 3 * max_peaks + 2  # After baseline and alpha
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


def get_spectral_index_location(model_name: str, max_peaks: int) -> Optional[int]:
    """
    Get the index of spectral index (alpha) parameter in theta.

    Args:
        model_name: Name of the model
        max_peaks: Maximum number of peaks

    Returns:
        Index of alpha parameter or None if not a 2D model
    """
    if "2d" not in model_name:
        return None

    if model_name == "emg_2d":
        return 4 * max_peaks  # After A, tau, u, w
    elif model_name == "exponential_2d":
        return 3 * max_peaks  # After A, tau, u
    elif model_name == "emg_2d_with_baseline":
        return 4 * max_peaks + 1  # After A, tau, u, w, baseline
    elif model_name == "exponential_2d_with_baseline":
        return 3 * max_peaks + 1  # After A, tau, u, baseline
    else:
        return None


def chebyshev_basis(freq: jnp.ndarray, k: int, freq_min: float, freq_max: float) -> jnp.ndarray:
    """
    Compute Chebyshev polynomial of order k for given frequency array.
    Not JIT-compiled due to dynamic control flow based on k.

    Args:
        freq: Frequency array
        k: Order of Chebyshev polynomial (0, 1, 2, ...)
        freq_min: Minimum frequency for normalization
        freq_max: Maximum frequency for normalization

    Returns:
        Chebyshev polynomial T_k evaluated at normalized frequencies
    """
    # Map frequency to [-1, 1]
    x = 2 * (freq - freq_min) / (freq_max - freq_min) - 1

    # Use recursive computation for numerical stability
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)

    if k == 0:
        return jnp.ones_like(x)
    elif k == 1:
        return x
    else:
        # Use iterative computation for k >= 2
        T_prev2 = jnp.ones_like(x)  # T_0
        T_prev1 = x                  # T_1

        for i in range(2, k + 1):
            T_current = 2 * x * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_current

        return T_prev1


def emg_model_3d_basis(t: jnp.ndarray, freq: jnp.ndarray, theta: jnp.ndarray,
                       max_peaks: int, n_basis: int, fit_pulses: bool = False,
                       ref_freq: float = 1400.0) -> jnp.ndarray:
    """
    3D EMG model using basis function decomposition with shared spectrum.
    All pulses share the same spectral shape modeled by basis functions.

    Args:
        t: Time array
        freq: Frequency array (MHz)
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un, w1,...,wn,
                               c0,...,c{K-1}, baseline, sigma, (Npulse)]
               where c_k are the basis function coefficients
        max_peaks: Maximum number of peaks
        n_basis: Number of basis functions (K)
        fit_pulses: Whether Npulse is included in theta
        ref_freq: Reference frequency for amplitude scaling

    Returns:
        3D model prediction (freq, time)
    """
    # Extract temporal parameters
    n_temporal = 4 * max_peaks  # A, tau, u, w for each peak
    temporal_params = theta[:n_temporal]

    # Extract spectral coefficients (shared for all pulses)
    spectral_coeffs = theta[n_temporal:n_temporal + n_basis]

    # Extract baseline
    baseline_idx = n_temporal + n_basis
    baseline = theta[baseline_idx]

    # Extract sigma
    sigma_idx = baseline_idx + 1
    # sigma = theta[sigma_idx]  # Not used in model generation

    # Optional: Npulse parameter
    if fit_pulses:
        npulse = jnp.round(theta[-1])
        npulse = jnp.clip(npulse, 1, max_peaks)
    else:
        npulse = max_peaks

    # Compute temporal model (1D) at reference frequency
    A = temporal_params[:max_peaks]
    tau = temporal_params[max_peaks:2*max_peaks]
    u = temporal_params[2*max_peaks:3*max_peaks]
    w = temporal_params[3*max_peaks:4*max_peaks]

    temporal_model = jnp.zeros_like(t)
    for i in range(max_peaks):
        pulse_active = i < npulse
        contribution = emg_pulse(t, A[i], tau[i], u[i], w[i])
        temporal_model += jnp.where(pulse_active, contribution, 0.0)

    # Compute spectral model using Chebyshev basis
    freq_min, freq_max = jnp.min(freq), jnp.max(freq)
    spectral_model = jnp.zeros_like(freq)

    for k in range(n_basis):
        basis_k = chebyshev_basis(freq, k, freq_min, freq_max)
        spectral_model += spectral_coeffs[k] * basis_k

    # Ensure positive spectral scaling using exponential
    # This prevents negative values and provides smooth variation
    spectral_model = jnp.exp(spectral_model)

    # Normalize spectral model at reference frequency for interpretability
    # Find closest frequency to ref_freq
    ref_idx = jnp.argmin(jnp.abs(freq - ref_freq))
    spectral_model = spectral_model / spectral_model[ref_idx]

    # Combine: outer product gives (freq, time) array
    model_3d = spectral_model[:, None] * temporal_model[None, :] + baseline

    return model_3d


def exponential_model_3d_basis(t: jnp.ndarray, freq: jnp.ndarray, theta: jnp.ndarray,
                               max_peaks: int, n_basis: int, fit_pulses: bool = False,
                               ref_freq: float = 1400.0) -> jnp.ndarray:
    """
    3D exponential model using basis function decomposition with shared spectrum.
    All pulses share the same spectral shape modeled by basis functions.

    Args:
        t: Time array
        freq: Frequency array (MHz)
        theta: Parameter array [A1,...,An, tau1,...,taun, u1,...,un,
                               c0,...,c{K-1}, baseline, sigma, (Npulse)]
               where c_k are the basis function coefficients
        max_peaks: Maximum number of peaks
        n_basis: Number of basis functions (K)
        fit_pulses: Whether Npulse is included in theta
        ref_freq: Reference frequency for amplitude scaling

    Returns:
        3D model prediction (freq, time)
    """
    # Extract temporal parameters
    n_temporal = 3 * max_peaks  # A, tau, u for each peak
    temporal_params = theta[:n_temporal]

    # Extract spectral coefficients (shared for all pulses)
    spectral_coeffs = theta[n_temporal:n_temporal + n_basis]

    # Extract baseline
    baseline_idx = n_temporal + n_basis
    baseline = theta[baseline_idx]

    # Extract sigma
    sigma_idx = baseline_idx + 1
    # sigma = theta[sigma_idx]  # Not used in model generation

    # Optional: Npulse parameter
    if fit_pulses:
        npulse = jnp.round(theta[-1])
        npulse = jnp.clip(npulse, 1, max_peaks)
    else:
        npulse = max_peaks

    # Compute temporal model (1D) at reference frequency
    A = temporal_params[:max_peaks]
    tau = temporal_params[max_peaks:2*max_peaks]
    u = temporal_params[2*max_peaks:3*max_peaks]

    temporal_model = jnp.zeros_like(t)
    for i in range(max_peaks):
        pulse_active = i < npulse
        contribution = exponential_pulse(t, A[i], tau[i], u[i])
        temporal_model += jnp.where(pulse_active, contribution, 0.0)

    # Compute spectral model using Chebyshev basis
    freq_min, freq_max = jnp.min(freq), jnp.max(freq)
    spectral_model = jnp.zeros_like(freq)

    for k in range(n_basis):
        basis_k = chebyshev_basis(freq, k, freq_min, freq_max)
        spectral_model += spectral_coeffs[k] * basis_k

    # Ensure positive spectral scaling using exponential
    spectral_model = jnp.exp(spectral_model)

    # Normalize spectral model at reference frequency
    ref_idx = jnp.argmin(jnp.abs(freq - ref_freq))
    spectral_model = spectral_model / spectral_model[ref_idx]

    # Combine: outer product gives (freq, time) array
    model_3d = spectral_model[:, None] * temporal_model[None, :] + baseline

    return model_3d