"""
Test with real FRB data from HDF5 file.
"""
import os
import sys
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import anesthetic
import h5py
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import exponential_model_with_baseline, get_model_function, get_param_names
from frbayes_jax.sampling import run_nested_sampling


def downsample(data, factor_time, factor_freq):
    """
    Downsample the waterfall data by averaging over blocks of size
    factor_time x factor_freq.
    
    Args:
        data: 2D array of shape (freq_bins, time_bins)
        factor_time: Downsampling factor in time dimension
        factor_freq: Downsampling factor in frequency dimension
    
    Returns:
        Downsampled data array
    """
    # Calculate the new shape
    new_freq_bins = data.shape[0] // factor_freq
    new_time_bins = data.shape[1] // factor_time
    
    # Reshape and average
    reshaped = data[:new_freq_bins * factor_freq, :new_time_bins * factor_time]
    reshaped = reshaped.reshape(new_freq_bins, factor_freq, new_time_bins, factor_time)
    downsampled = np.nanmean(reshaped, axis=(1, 3))
    
    return downsampled


def load_real_data(data_file, preprocessing_mode="default"):
    """
    Load and preprocess real FRB data from HDF5 file.
    
    Args:
        data_file: Path to the HDF5 data file
        preprocessing_mode: One of "default", "paper", or "raw"
    
    Returns:
        time_axis: Time array
        pulse_profile: 1D pulse profile (S/N or intensity)
    """
    # Constants from the frbayes_cpu settings
    original_freq_res = 24414.0625  # Hz
    original_time_res = 0.98304e-3  # s
    desired_freq_res = 3.125e6  # Hz
    desired_time_res = 7.86432e-3  # s
    
    # Load data from HDF5 file
    with h5py.File(data_file, "r") as f:
        wfall = f["waterfall"][:]
    
    if preprocessing_mode == "raw":
        # No NaN replacement, no downsampling
        final_wfall = wfall
        final_time_res = original_time_res
        pulse_profile = np.nanmean(final_wfall, axis=0)
    
    elif preprocessing_mode == "paper":
        # Paper preprocessing: NaN replacement and downsampling
        # Replace NaN values with median of off-burst region (first 10%)
        off_burst_time_bins = int(wfall.shape[1] * 0.1)
        off_burst_data = wfall[:, :off_burst_time_bins]
        off_burst_median = np.nanmedian(off_burst_data)
        wfall[np.isnan(wfall)] = off_burst_median
        
        # Calculate downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)
        
        # Downsample
        wfall_downsampled = downsample(wfall, factor_time, factor_freq)
        final_wfall = wfall_downsampled
        final_time_res = desired_time_res
        
        # Calculate S/N profile
        pulse_profile_intensity = np.nanmean(final_wfall, axis=0)
        pulse_profile_intensity = np.atleast_1d(pulse_profile_intensity)
        
        # Baseline estimation from off-pulse region
        num_time_bins = pulse_profile_intensity.shape[0]
        off_pulse_bins = int(num_time_bins * 0.1)
        baseline = np.nanmedian(pulse_profile_intensity[:off_pulse_bins])
        
        # Subtract baseline
        profile_baseline_subtracted = pulse_profile_intensity - baseline
        
        # Estimate noise RMS
        noise_std = np.nanstd(profile_baseline_subtracted[:off_pulse_bins])
        if np.isnan(noise_std) or noise_std == 0:
            noise_std = 1e-9
        
        # Calculate S/N profile
        pulse_profile = profile_baseline_subtracted / noise_std
    
    else:  # default
        # Default preprocessing: Replace NaN with 0 and downsample
        wfall[np.isnan(wfall)] = 0
        
        # Calculate downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)
        
        # Downsample
        wfall_downsampled = downsample(wfall, factor_time, factor_freq)
        final_wfall = wfall_downsampled
        final_time_res = desired_time_res
        
        # Calculate mean profile
        pulse_profile = np.mean(wfall_downsampled, axis=0)
    
    # Ensure pulse_profile is 1D
    pulse_profile = np.atleast_1d(pulse_profile)
    
    # Generate time axis
    num_time_bins = len(pulse_profile)
    time_axis = np.arange(num_time_bins) * final_time_res  # Time in seconds
    
    return time_axis, pulse_profile


def main():
    """
    Test nested sampling with real FRB data.
    """
    print("="*60)
    print("TEST: REAL FRB DATA ANALYSIS")
    print("="*60)
    
    # Settings
    model_name = "exponential_with_baseline"
    max_peaks = 10  # Increased to 10 peaks for real data
    fit_pulses = False  # Fixed number of pulses initially
    
    # Data file path
    data_file = "data_frbayes/20191221A_original_data.h5"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        return
    
    # Load and preprocess real data
    print("\nLoading real FRB data...")
    print(f"  Data file: {data_file}")
    print(f"  Preprocessing mode: default")
    
    t_np, data_np = load_real_data(data_file, preprocessing_mode="default")
    
    print(f"  Data shape: {data_np.shape}")
    print(f"  Time range: [{t_np[0]:.3f}, {t_np[-1]:.3f}] seconds")
    print(f"  Data range: [{np.min(data_np):.3f}, {np.max(data_np):.3f}]")
    
    # Don't normalize - use raw intensity data so baseline fitting makes sense
    # The baseline will capture the actual offset in the data
    data_for_fitting = data_np
    
    # Plot the real data
    plt.figure(figsize=(12, 5))
    plt.plot(t_np, data_for_fitting, 'k-', linewidth=0.5, alpha=0.8, label='Real FRB data')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Intensity')
    plt.title('Real FRB Data: 20191221A')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_real_data_input.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nData plot saved to test_real_data_input.png")
    
    # Set up prior bounds based on the data characteristics
    # For exponential_with_baseline model
    data_range = np.max(data_for_fitting) - np.min(data_for_fitting)
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': data_range},  # Amplitude relative to data range
        'tau': {'min': 0.01, 'max': 2.0},  # Tau for exponential decay
        'u': {'min': 0.0, 'max': t_np[-1]},  # Arrival time covers full data range
        'baseline': {'min': np.min(data_for_fitting) - 0.1*data_range, 
                     'max': np.max(data_for_fitting)},  # Baseline can be anywhere in data range
        'log_sigma': {'min': jnp.log(0.0001), 'max': jnp.log(0.1*data_range)}  # Log-uniform for sigma
    }
    
    # Run nested sampling
    print("\nRunning nested sampling...")
    print(f"  Model: {model_name}")
    print(f"  Max peaks: {max_peaks}")
    print(f"  Fit pulses: {fit_pulses}")
    
    # Calculate nested sampling parameters
    # exponential_with_baseline: 3 params per peak (A, tau, u) + baseline + sigma
    if fit_pulses:
        ndims = 3 * max_peaks + 3  # 3 params per peak + baseline + sigma + n_pulses
    else:
        ndims = 3 * max_peaks + 2  # 3 params per peak + baseline + sigma
    
    num_live_points = ndims * 25
    num_delete = num_live_points // 2
    num_inner_steps = ndims * 5
    
    print(f"  Number of dimensions: {ndims}")
    print(f"  Number of live points: {num_live_points}")
    
    # Run nested sampling
    final_state = run_nested_sampling(
        model_name=model_name,
        data=data_for_fitting,
        t=t_np,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        num_live_points=num_live_points,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
        log_tolerance=-3.0,
        seed=42
    )
    
    print("\nNested sampling completed.")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/results_real_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter names
    param_names = get_param_names(model_name, max_peaks, fit_pulses)
    
    # Save chains in anesthetic format
    print("\nSaving chains in anesthetic format...")
    chain_path = os.path.join(output_dir, 'chains')
    os.makedirs(os.path.dirname(chain_path) if os.path.dirname(chain_path) else '.', exist_ok=True)
    
    # Save particles, logL, and logL_birth
    chain_data = np.column_stack([
        final_state.particles,
        final_state.loglikelihood,
        final_state.loglikelihood_birth
    ])
    np.savetxt(f"{chain_path}_dead-birth.txt", chain_data)
    print(f"Chains saved to {chain_path}_dead-birth.txt")
    
    # Create NestedSamples object for analysis
    nested_samples = anesthetic.NestedSamples(
        data=final_state.particles,
        logL=final_state.loglikelihood,
        logL_birth=final_state.loglikelihood_birth,
    )
    
    # Get posterior statistics
    best_fit = nested_samples.mean().values
    std_fit = nested_samples.std().values
    
    print("\nBest-fit parameters (posterior mean ± std):")
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_fit[i]:.4f} ± {std_fit[i]:.4f}")
    
    # Plot best-fit model
    model_func = get_model_function(model_name)
    
    plt.figure(figsize=(12, 5))
    plt.plot(t_np, data_for_fitting, 'k-', linewidth=0.5, alpha=0.8, label='Real data')
    
    # Best-fit model
    best_fit_model = model_func(jnp.array(t_np), jnp.array(best_fit), max_peaks, fit_pulses)
    plt.plot(t_np, np.array(best_fit_model), 'r-', linewidth=2, alpha=0.7, label='Best-fit model')
    
    # Plot individual components if we have multiple peaks
    if max_peaks > 1 and not fit_pulses:
        # Extract baseline value
        baseline_value = best_fit[3*max_peaks]  # Baseline is after all A, tau, u params
        
        for i in range(max_peaks):
            # Create parameters for single peak with baseline
            single_peak_params = np.zeros(5)  # A, tau, u, baseline, sigma
            single_peak_params[0] = best_fit[i]  # Amplitude
            single_peak_params[1] = best_fit[max_peaks + i]  # Tau
            single_peak_params[2] = best_fit[2*max_peaks + i]  # u
            single_peak_params[3] = 0  # No baseline for component visualization
            single_peak_params[4] = 0  # No noise for component plot
            
            # Use exponential_with_baseline model for single component
            component = get_model_function("exponential_with_baseline")(
                jnp.array(t_np), jnp.array(single_peak_params), 1, False
            )
            plt.plot(t_np, np.array(component) + baseline_value, '--', linewidth=1, alpha=0.5, 
                    label=f'Component {i+1}')
        
        # Plot baseline
        plt.axhline(y=baseline_value, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Intensity')
    plt.title('Real FRB Data: Model Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'model_fit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nModel fit plot saved to {output_dir}/model_fit.png")
    
    # Save parameter names
    with open(os.path.join(output_dir, 'param_names.txt'), 'w') as f:
        for name in param_names:
            f.write(f"{name}\n")
    print(f"Parameter names saved to {output_dir}/param_names.txt")
    
    # Save metadata
    metadata = {
        'data_file': data_file,
        'preprocessing_mode': 'default',
        'model_name': model_name,
        'max_peaks': max_peaks,
        'fit_pulses': fit_pulses,
        'num_params': len(param_names),
        'num_samples': len(final_state.particles),
        'data_range': {
            'min': float(np.min(data_for_fitting)),
            'max': float(np.max(data_for_fitting))
        }
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {output_dir}/metadata.json")
    
    # Calculate and print evidence
    log_Z = nested_samples.logZ()
    print(f"\nLog evidence: {log_Z:.2f}")
    
    print("\n" + "="*60)
    print("REAL DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()