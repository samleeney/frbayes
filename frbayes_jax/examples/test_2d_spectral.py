"""
Test 2D fitting with spectral index on simulated or real FRB data.
"""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import exponential_model_2d, get_param_names, get_spectral_index_location
from frbayes_jax.sampling import run_nested_sampling_2d
from frbayes_jax.data import preprocess_data_2d
from frbayes_jax.analysis import plot_corner
import anesthetic


def simulate_2d_data(n_freq=32, n_time=500, max_peaks=2, spectral_index=-1.4, seed=42):
    """
    Simulate 2D FRB data with known spectral index.
    
    Args:
        n_freq: Number of frequency channels
        n_time: Number of time bins
        max_peaks: Number of pulses
        spectral_index: True spectral index
        seed: Random seed
    
    Returns:
        data_2d, time_axis, freq_axis, true_params
    """
    # Create axes
    time_axis = np.linspace(0, 4, n_time)
    freq_axis = np.linspace(1200, 1600, n_freq)  # MHz
    
    # Set true parameters for exponential_2d model
    true_params = []
    
    # Amplitudes
    amplitudes = [5.0, 3.0] if max_peaks >= 2 else [5.0]
    for i in range(max_peaks):
        true_params.append(amplitudes[i] if i < len(amplitudes) else 2.0)
    
    # Decay times
    taus = [0.1, 0.15] if max_peaks >= 2 else [0.1]
    for i in range(max_peaks):
        true_params.append(taus[i] if i < len(taus) else 0.1)
    
    # Arrival times (sorted)
    arrivals = [1.0, 2.5] if max_peaks >= 2 else [1.5]
    for i in range(max_peaks):
        true_params.append(arrivals[i] if i < len(arrivals) else 1.0 + i)
    
    # Spectral index
    true_params.append(spectral_index)
    
    # Noise level
    sigma = 0.5
    true_params.append(sigma)
    
    true_params = jnp.array(true_params)
    
    # Generate model
    model_2d = exponential_model_2d(
        jnp.array(time_axis), 
        jnp.array(freq_axis), 
        true_params, 
        max_peaks, 
        fit_pulses=False
    )
    
    # Add noise
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, shape=model_2d.shape) * sigma
    data_2d = np.array(model_2d + noise)
    
    return data_2d, time_axis, freq_axis, true_params


def test_real_data_2d():
    """Test 2D fitting on real FRB data."""
    
    # Check if data file exists
    data_file = "../data_frbayes/frb121102_width_4e-05_dm_560.42.h5"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Using simulated data instead...")
        return test_simulated_data_2d()
    
    # Load and preprocess 2D data
    print("Loading real FRB data for 2D fitting...")
    wfall_2d, time_axis, freq_axis, noise_per_channel = preprocess_data_2d(
        data_file=data_file,
        original_freq_res=122070.3125,  # Hz
        original_time_res=4e-05,         # seconds
        desired_freq_res=122070.3125*4,  # Downsample by 4 in frequency
        desired_time_res=4e-05*4,        # Downsample by 4 in time
        freq_min=1200.0,                 # MHz
        freq_max=1600.0,                 # MHz
        preprocessing_mode="default"
    )
    
    print(f"Data shape: {wfall_2d.shape} (freq × time)")
    print(f"Frequency range: {freq_axis.min():.1f} - {freq_axis.max():.1f} MHz")
    print(f"Time range: {time_axis.min():.3f} - {time_axis.max():.3f} s")
    
    # Set prior bounds with correct time range
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': np.max(wfall_2d)},
        'tau': {'min': 0.001, 'max': 1.0},
        'u': {'min': time_axis.min(), 'max': time_axis.max()},  # Use actual time range
        'width': {'min': 0.001, 'max': 1.0},
        'log_sigma': {'min': jnp.log(0.001), 'max': jnp.log(np.std(wfall_2d))},
        'spectral_index': {'min': -3.0, 'max': 1.0}
    }
    
    # Run 2D nested sampling
    print("\nRunning 2D nested sampling with spectral index...")
    results = run_nested_sampling_2d(
        model_name="exponential_2d",
        data_2d=wfall_2d,
        t=time_axis,
        freq=freq_axis,
        noise_per_channel=noise_per_channel,
        prior_bounds=prior_bounds,
        max_peaks=2,
        fit_pulses=False,
        num_live_points=500,
        num_delete=25,
        num_inner_steps=20,
        log_tolerance=-3.0,
        seed=42
    )
    
    return results, wfall_2d, time_axis, freq_axis


def test_simulated_data_2d():
    """Test 2D fitting on simulated data with known spectral index."""
    
    print("Generating simulated 2D FRB data...")
    true_spectral_index = -1.4
    data_2d, time_axis, freq_axis, true_params = simulate_2d_data(
        n_freq=32,
        n_time=500,
        max_peaks=2,
        spectral_index=true_spectral_index,
        seed=42
    )
    
    print(f"Data shape: {data_2d.shape} (freq × time)")
    print(f"True spectral index: {true_spectral_index}")
    print(f"True parameters: {true_params}")
    
    # Estimate noise per channel
    off_pulse_bins = int(len(time_axis) * 0.1)
    noise_per_channel = np.std(data_2d[:, :off_pulse_bins], axis=1)
    
    # Set prior bounds with correct time range
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 10.0},
        'tau': {'min': 0.001, 'max': 1.0},
        'u': {'min': time_axis.min(), 'max': time_axis.max()},  # Use actual time range
        'width': {'min': 0.001, 'max': 1.0},
        'log_sigma': {'min': jnp.log(0.001), 'max': jnp.log(1.0)},
        'spectral_index': {'min': -3.0, 'max': 1.0}
    }
    
    # Run 2D nested sampling
    print("\nRunning 2D nested sampling with spectral index...")
    results = run_nested_sampling_2d(
        model_name="exponential_2d",
        data_2d=data_2d,
        t=time_axis,
        freq=freq_axis,
        noise_per_channel=noise_per_channel,
        prior_bounds=prior_bounds,
        max_peaks=2,
        fit_pulses=False,
        num_live_points=500,
        num_delete=25,
        num_inner_steps=20,
        log_tolerance=-3.0,
        seed=42
    )
    
    # Create NestedSamples object for analysis (following test_2pulses_exponential_fitted.py pattern)
    param_names = get_param_names("exponential_2d", max_peaks=2, fit_pulses=False)
    nested_samples = anesthetic.NestedSamples(
        data=results.particles,
        logL=results.loglikelihood,
        logL_birth=results.loglikelihood_birth,
        columns=param_names
    )
    
    # Extract spectral index results
    alpha_idx = get_spectral_index_location("exponential_2d", max_peaks=2)
    if alpha_idx is not None:
        # Get spectral index statistics using anesthetic
        alpha_mean = nested_samples.iloc[:, alpha_idx].mean()
        alpha_std = nested_samples.iloc[:, alpha_idx].std()
        
        print(f"\nSpectral Index Results:")
        print(f"  True value: {true_spectral_index}")
        print(f"  Fitted: {alpha_mean:.3f} ± {alpha_std:.3f}")
        print(f"  Deviation: {abs(alpha_mean - true_spectral_index)/alpha_std:.1f} sigma")
    
    return results, data_2d, time_axis, freq_axis


def plot_2d_results(results, data_2d, time_axis, freq_axis, model_name="exponential_2d", max_peaks=2):
    """Plot results from 2D fitting."""
    
    # Create NestedSamples object for analysis
    param_names = get_param_names(model_name, max_peaks, fit_pulses=False)
    nested_samples = anesthetic.NestedSamples(
        data=results.particles,
        logL=results.loglikelihood,
        logL_birth=results.loglikelihood_birth,
        columns=param_names
    )
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. Original 2D data
    ax = axes[0, 0]
    im = ax.imshow(data_2d, aspect='auto', origin='lower', 
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Original Data')
    plt.colorbar(im, ax=ax)
    
    # 2. Frequency-averaged profile
    ax = axes[0, 1]
    profile_1d = np.mean(data_2d, axis=0)
    ax.plot(time_axis, profile_1d)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity')
    ax.set_title('Frequency-Averaged Profile')
    ax.grid(True, alpha=0.3)
    
    # 3. Spectrum (peak intensity vs frequency)
    ax = axes[0, 2]
    peak_intensity = np.max(data_2d, axis=1)
    ax.plot(freq_axis, peak_intensity, 'o-')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Peak Intensity')
    ax.set_title('Peak Intensity Spectrum')
    ax.grid(True, alpha=0.3)
    
    # Get best-fit parameters
    # Get MAP estimate (Maximum A Posteriori)
    best_idx = np.argmax(results.loglikelihood)
    best_params = results.particles[best_idx]
    
    # 4. Best-fit model
    from frbayes_jax.models import get_model_function
    model_func = get_model_function(model_name)
    model_2d = model_func(
        jnp.array(time_axis),
        jnp.array(freq_axis),
        best_params,
        max_peaks,
        fit_pulses=False
    )
    
    ax = axes[1, 0]
    im = ax.imshow(np.array(model_2d), aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Best-fit Model')
    plt.colorbar(im, ax=ax)
    
    # 5. Residuals
    ax = axes[1, 1]
    residuals = data_2d - np.array(model_2d)
    im = ax.imshow(residuals, aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                   cmap='RdBu_r', vmin=-3, vmax=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Residuals')
    plt.colorbar(im, ax=ax)
    
    # 6. Spectral index posterior
    ax = axes[1, 2]
    alpha_idx = get_spectral_index_location(model_name, max_peaks)
    if alpha_idx is not None:
        # Use nested_samples to get spectral index
        alpha_samples = nested_samples.iloc[:, alpha_idx]
        alpha_mean = alpha_samples.mean()
        alpha_std = alpha_samples.std()
        
        ax.hist(alpha_samples, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(alpha_mean, color='red', linestyle='--', 
                   label=f'α = {alpha_mean:.2f} ± {alpha_std:.2f}')
        ax.set_xlabel('Spectral Index α')
        ax.set_ylabel('Posterior Samples')
        ax.set_title('Spectral Index Posterior')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('2D FRB Fitting with Spectral Index', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/2d_fitting_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: results/2d_fitting_{timestamp}.png")
    
    plt.show()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Test with real data if available, otherwise use simulated
    print("="*60)
    print("Testing 2D FRB Fitting with Spectral Index")
    print("="*60)
    
    # Run test (will use real data if available, otherwise simulated)
    results, data_2d, time_axis, freq_axis = test_real_data_2d()
    
    # Plot results
    plot_2d_results(results, data_2d, time_axis, freq_axis, 
                    model_name="exponential_2d", max_peaks=2)
    
    # Print parameter summary
    print("\n" + "="*60)
    print("Parameter Estimates:")
    print("="*60)
    
    param_names = get_param_names("exponential_2d", max_peaks=2, fit_pulses=False)
    
    # Create NestedSamples object like in test_2pulses_exponential_fitted.py
    nested_samples = anesthetic.NestedSamples(
        data=results.particles,
        logL=results.loglikelihood,
        logL_birth=results.loglikelihood_birth,
        columns=param_names
    )
    
    # Get posterior statistics from anesthetic
    best_fit = nested_samples.mean().values
    std_fit = nested_samples.std().values
    
    for i, name in enumerate(param_names):
        print(f"{name}: {best_fit[i]:.3f} ± {std_fit[i]:.3f}")
    
    print("\nDone!")