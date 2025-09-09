"""
Data preprocessing and simulation utilities for FRBayes JAX.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional
import h5py


def downsample(data: np.ndarray, factor_time: int, factor_freq: int) -> np.ndarray:
    """
    Downsample 2D data array by averaging.
    
    Args:
        data: 2D array (freq, time)
        factor_time: Downsampling factor in time dimension
        factor_freq: Downsampling factor in frequency dimension
    
    Returns:
        Downsampled array
    """
    return data.reshape(
        data.shape[0] // factor_freq,
        factor_freq,
        data.shape[1] // factor_time,
        factor_time,
    ).mean(axis=(1, 3))


def preprocess_data(
    data_file: str,
    original_freq_res: float,
    original_time_res: float,
    desired_freq_res: float,
    desired_time_res: float,
    freq_min: float,
    freq_max: float,
    preprocessing_mode: str = "default"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess FRB data from HDF5 file.
    
    Args:
        data_file: Path to HDF5 data file
        original_freq_res: Original frequency resolution in Hz
        original_time_res: Original time resolution in seconds
        desired_freq_res: Desired frequency resolution in Hz
        desired_time_res: Desired time resolution in seconds
        freq_min: Minimum frequency in MHz
        freq_max: Maximum frequency in MHz
        preprocessing_mode: "default", "paper", or "raw"
    
    Returns:
        Tuple of (downsampled_wfall, pulse_profile_snr, time_axis)
    """
    # Load data from HDF5 file
    with h5py.File(data_file, 'r') as f:
        wfall = f['waterfall'][:]
    
    if preprocessing_mode == "raw":
        # No NaN replacement, no downsampling
        final_wfall_output = wfall
        final_time_res_for_axis = original_time_res
        pulse_profile_snr = np.nanmean(final_wfall_output, axis=0)
        
    elif preprocessing_mode == "paper":
        # RFI Mitigation: Replace NaN with off-burst median
        off_burst_time_bins = int(wfall.shape[1] * 0.1)
        off_burst_data = wfall[:, :off_burst_time_bins]
        off_burst_median = np.nanmedian(off_burst_data)
        wfall[np.isnan(wfall)] = off_burst_median
        
        # Calculate downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)
        
        # Downsample
        wfall_downsampled = downsample(wfall, factor_time, factor_freq)
        final_wfall_output = wfall_downsampled
        final_time_res_for_axis = desired_time_res
        
        # S/N Calculation for paper preprocessing
        pulse_profile_intensity = np.nanmean(final_wfall_output, axis=0)
        pulse_profile_intensity = np.atleast_1d(pulse_profile_intensity)
        
        # Baseline subtraction
        num_time_bins_profile = pulse_profile_intensity.shape[0]
        off_pulse_bins_1d = int(num_time_bins_profile * 0.1)
        baseline_1d = np.nanmedian(pulse_profile_intensity[:off_pulse_bins_1d])
        profile_baseline_subtracted = pulse_profile_intensity - baseline_1d
        
        # Noise estimation
        noise_std_1d = np.nanstd(profile_baseline_subtracted[:off_pulse_bins_1d])
        epsilon = 1e-9
        if np.isnan(noise_std_1d) or noise_std_1d == 0:
            noise_std_1d = epsilon
        
        # Calculate S/N
        pulse_profile_snr = profile_baseline_subtracted / noise_std_1d
        
    else:  # default
        # Replace NaN with 0
        wfall[np.isnan(wfall)] = 0
        
        # Calculate downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)
        
        # Downsample
        wfall_downsampled = downsample(wfall, factor_time, factor_freq)
        final_wfall_output = wfall_downsampled
        final_time_res_for_axis = desired_time_res
        
        # Calculate pulse profile as mean
        pulse_profile_snr = np.mean(wfall_downsampled, axis=0)
    
    # Ensure pulse_profile_snr is 1D
    pulse_profile_snr = np.atleast_1d(pulse_profile_snr)
    
    # Generate time axis
    num_freq_bins, num_time_bins = final_wfall_output.shape
    time_axis = np.arange(num_time_bins) * final_time_res_for_axis
    
    return final_wfall_output, pulse_profile_snr, time_axis


def simulate_frb_data(
    model_func: callable,
    theta: jnp.ndarray,
    max_peaks: int,
    fit_pulses: bool,
    t_min: float = 0.0,
    t_max: float = 4.0,
    num_points: int = 500,
    add_noise: bool = True,
    seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate FRB data using a given model.
    
    Args:
        model_func: Model function (e.g., emg_model)
        theta: Model parameters
        max_peaks: Maximum number of peaks
        fit_pulses: Whether Npulse is in theta
        t_min: Start time
        t_max: End time
        num_points: Number of time points
        add_noise: Whether to add Gaussian noise
        seed: Random seed for noise
    
    Returns:
        Tuple of (time_axis, pulse_profile)
    """
    # Generate time axis
    t = jnp.linspace(t_min, t_max, num_points)
    
    # Generate model prediction
    model_pred = model_func(t, theta, max_peaks, fit_pulses)
    
    # Add noise if requested
    if add_noise:
        # Get sigma from theta
        if "emg" in model_func.__name__:
            if "baseline" in model_func.__name__:
                sigma_idx = 4 * max_peaks + 1
            else:
                sigma_idx = 4 * max_peaks
        else:  # exponential
            if "baseline" in model_func.__name__:
                sigma_idx = 3 * max_peaks + 1
            else:
                sigma_idx = 3 * max_peaks
        
        sigma = theta[sigma_idx]
        
        # Generate noise
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, shape=t.shape) * sigma
        pulse_profile = model_pred + noise
    else:
        pulse_profile = model_pred
    
    return t, pulse_profile