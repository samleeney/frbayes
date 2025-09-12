"""
Test 2D fitting with spectral index and RFI mitigation using Bayesian anomaly detection.
Simulates FRB data with 3 peaks and adds RFI as constant spikes at fixed frequencies.
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

from frbayes_jax.models import exponential_model_2d, get_param_names, get_spectral_index_location, get_npulse_index
from frbayes_jax.sampling import run_nested_sampling_2d
import anesthetic


def simulate_2d_data_with_rfi(n_freq=32, n_time=500, spectral_index=-1.4, 
                              rfi_channels=None, rfi_amplitude=20.0, seed=42):
    """
    Simulate 2D FRB data with 3 peaks, known spectral index, and RFI.
    
    Args:
        n_freq: Number of frequency channels
        n_time: Number of time bins
        spectral_index: True spectral index
        rfi_channels: List of channel indices for RFI (if None, uses [5, 10, 20])
        rfi_amplitude: Amplitude of RFI spikes
        seed: Random seed
    
    Returns:
        data_2d, time_axis, freq_axis, true_params, rfi_mask
    """
    # Create axes
    time_axis = np.linspace(0, 4, n_time)
    freq_axis = np.linspace(1200, 1600, n_freq)  # MHz
    
    # Set true parameters for exponential_2d model with 3 peaks
    true_params = []
    
    # Amplitudes for 3 peaks
    amplitudes = [5.0, 3.0, 2.0]
    for amp in amplitudes:
        true_params.append(amp)
    
    # Decay times for 3 peaks
    taus = [0.1, 0.15, 0.12]
    for tau in taus:
        true_params.append(tau)
    
    # Arrival times (sorted) for 3 peaks
    arrivals = [0.8, 1.8, 3.0]
    for u in arrivals:
        true_params.append(u)
    
    # Spectral index
    true_params.append(spectral_index)
    
    # Noise level
    sigma = 0.5
    true_params.append(sigma)
    
    true_params = jnp.array(true_params)
    
    # Generate model (using 3 peaks, no Npulse parameter)
    model_2d = exponential_model_2d(
        jnp.array(time_axis), 
        jnp.array(freq_axis), 
        true_params, 
        max_peaks=3, 
        fit_pulses=False
    )
    
    # Add noise
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, shape=model_2d.shape) * sigma
    data_2d = np.array(model_2d + noise)
    
    # Add RFI: constant spikes at specific frequency channels
    if rfi_channels is None:
        rfi_channels = [5, 10, 20]  # Default RFI channels
    
    rfi_mask = np.zeros(n_freq, dtype=bool)
    for ch in rfi_channels:
        if ch < n_freq:
            # Add constant RFI across all time bins for this channel
            data_2d[ch, :] += rfi_amplitude
            rfi_mask[ch] = True
    
    print(f"Added RFI to channels: {rfi_channels} with amplitude {rfi_amplitude}")
    
    return data_2d, time_axis, freq_axis, true_params, rfi_mask, 3  # Return true number of peaks


def test_rfi_mitigation():
    """Test 2D fitting with and without RFI mitigation."""
    
    print("="*60)
    print("Testing RFI Mitigation with Bayesian Anomaly Detection")
    print("="*60)
    
    # Generate simulated data with RFI
    print("\nGenerating simulated 2D FRB data with 3 peaks and RFI...")
    true_spectral_index = -1.4
    data_2d, time_axis, freq_axis, true_params, rfi_mask, true_npulse = simulate_2d_data_with_rfi(
        n_freq=32,
        n_time=500,
        spectral_index=true_spectral_index,
        rfi_channels=[5, 10, 20],  # Add RFI to these channels
        rfi_amplitude=20.0,  # Strong RFI
        seed=42
    )
    
    print(f"Data shape: {data_2d.shape} (freq × time)")
    print(f"True spectral index: {true_spectral_index}")
    print(f"True number of pulses: {true_npulse}")
    print(f"RFI-affected channels: {np.where(rfi_mask)[0].tolist()}")
    
    # Estimate noise per channel (avoiding RFI channels)
    off_pulse_bins = int(len(time_axis) * 0.1)
    noise_per_channel = np.std(data_2d[:, :off_pulse_bins], axis=1)
    
    # Set prior bounds
    max_peaks = 5
    prior_bounds = {
        'amplitude': {'min': 0.001, 'max': 10.0},
        'tau': {'min': 0.001, 'max': 1.0},
        'u': {'min': time_axis.min(), 'max': time_axis.max()},
        'width': {'min': 0.001, 'max': 1.0},
        'log_sigma': {'min': jnp.log(0.001), 'max': jnp.log(1.0)},
        'spectral_index': {'min': -3.0, 'max': 1.0},
        'log_anomaly_prob': {'min': -10.0, 'max': -0.1}  # For RFI mitigation
    }
    
    # Run WITHOUT RFI mitigation
    print("\n" + "="*60)
    print("FITTING WITHOUT RFI MITIGATION")
    print("="*60)
    
    results_no_rfi = run_nested_sampling_2d(
        model_name="exponential_2d",
        data_2d=data_2d,
        t=time_axis,
        freq=freq_axis,
        noise_per_channel=noise_per_channel,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=True,
        use_rfi_mitigation=False,  # No RFI mitigation
        num_live_points=500,
        num_delete=25,
        num_inner_steps=20,
        log_tolerance=-3.0,
        seed=42
    )
    
    # Analyze results without RFI mitigation
    param_names_no_rfi = get_param_names("exponential_2d", max_peaks=max_peaks, fit_pulses=True)
    nested_samples_no_rfi = anesthetic.NestedSamples(
        data=results_no_rfi.particles,
        logL=results_no_rfi.loglikelihood,
        logL_birth=results_no_rfi.loglikelihood_birth,
        columns=param_names_no_rfi
    )
    
    # Extract results
    alpha_idx = get_spectral_index_location("exponential_2d", max_peaks)
    npulse_idx = get_npulse_index("exponential_2d", max_peaks, fit_pulses=True)
    
    if alpha_idx is not None:
        alpha_mean_no_rfi = nested_samples_no_rfi.iloc[:, alpha_idx].mean()
        alpha_std_no_rfi = nested_samples_no_rfi.iloc[:, alpha_idx].std()
        print(f"\nSpectral Index (No RFI mitigation):")
        print(f"  True value: {true_spectral_index}")
        print(f"  Fitted: {alpha_mean_no_rfi:.3f} ± {alpha_std_no_rfi:.3f}")
        print(f"  Error: {abs(alpha_mean_no_rfi - true_spectral_index):.3f}")
    
    if npulse_idx is not None:
        npulse_samples_no_rfi = results_no_rfi.particles[:, npulse_idx]
        npulse_mean_no_rfi = np.mean(npulse_samples_no_rfi)
        npulse_std_no_rfi = np.std(npulse_samples_no_rfi)
        print(f"\nNumber of Pulses (No RFI mitigation):")
        print(f"  True value: {true_npulse}")
        print(f"  Fitted: {npulse_mean_no_rfi:.2f} ± {npulse_std_no_rfi:.2f}")
    
    # Run WITH RFI mitigation
    print("\n" + "="*60)
    print("FITTING WITH RFI MITIGATION")
    print("="*60)
    
    results_with_rfi = run_nested_sampling_2d(
        model_name="exponential_2d",
        data_2d=data_2d,
        t=time_axis,
        freq=freq_axis,
        noise_per_channel=noise_per_channel,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=True,
        use_rfi_mitigation=True,  # Enable RFI mitigation
        num_live_points=500,
        num_delete=25,
        num_inner_steps=20,
        log_tolerance=-3.0,
        seed=43  # Different seed for variety
    )
    
    # Analyze results with RFI mitigation
    # Note: param names include extra log_p parameter when RFI mitigation is enabled
    param_names_with_rfi = get_param_names("exponential_2d", max_peaks=max_peaks, fit_pulses=True)
    param_names_with_rfi.append('log_p')  # Add anomaly probability parameter
    
    nested_samples_with_rfi = anesthetic.NestedSamples(
        data=results_with_rfi.particles,
        logL=results_with_rfi.loglikelihood,
        logL_birth=results_with_rfi.loglikelihood_birth,
        columns=param_names_with_rfi
    )
    
    if alpha_idx is not None:
        alpha_mean_with_rfi = nested_samples_with_rfi.iloc[:, alpha_idx].mean()
        alpha_std_with_rfi = nested_samples_with_rfi.iloc[:, alpha_idx].std()
        print(f"\nSpectral Index (With RFI mitigation):")
        print(f"  True value: {true_spectral_index}")
        print(f"  Fitted: {alpha_mean_with_rfi:.3f} ± {alpha_std_with_rfi:.3f}")
        print(f"  Error: {abs(alpha_mean_with_rfi - true_spectral_index):.3f}")
    
    if npulse_idx is not None:
        npulse_samples_with_rfi = results_with_rfi.particles[:, npulse_idx]
        npulse_mean_with_rfi = np.mean(npulse_samples_with_rfi)
        npulse_std_with_rfi = np.std(npulse_samples_with_rfi)
        print(f"\nNumber of Pulses (With RFI mitigation):")
        print(f"  True value: {true_npulse}")
        print(f"  Fitted: {npulse_mean_with_rfi:.2f} ± {npulse_std_with_rfi:.2f}")
    
    # Extract anomaly probability
    log_p_samples = nested_samples_with_rfi.iloc[:, -1]  # Last parameter is log_p
    p_mean = np.exp(log_p_samples.mean())
    p_std = np.exp(log_p_samples).std()
    print(f"\nAnomaly Probability:")
    print(f"  Mean p: {p_mean:.4f} ± {p_std:.4f}")
    print(f"  Expected anomaly fraction: {len(np.where(rfi_mask)[0]) / len(freq_axis):.3f}")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\nSpectral Index Recovery:")
    print(f"  True value: {true_spectral_index}")
    print(f"  Without RFI mitigation: {alpha_mean_no_rfi:.3f} ± {alpha_std_no_rfi:.3f} (error: {abs(alpha_mean_no_rfi - true_spectral_index):.3f})")
    print(f"  With RFI mitigation:    {alpha_mean_with_rfi:.3f} ± {alpha_std_with_rfi:.3f} (error: {abs(alpha_mean_with_rfi - true_spectral_index):.3f})")
    
    improvement = abs(alpha_mean_no_rfi - true_spectral_index) - abs(alpha_mean_with_rfi - true_spectral_index)
    print(f"  Improvement: {improvement:.3f} ({improvement/abs(alpha_mean_no_rfi - true_spectral_index)*100:.1f}% reduction in error)")
    
    print(f"\nNumber of Pulses Recovery:")
    print(f"  True value: {true_npulse}")
    print(f"  Without RFI mitigation: {npulse_mean_no_rfi:.2f} ± {npulse_std_no_rfi:.2f}")
    print(f"  With RFI mitigation:    {npulse_mean_with_rfi:.2f} ± {npulse_std_with_rfi:.2f}")
    
    return (results_no_rfi, results_with_rfi, data_2d, time_axis, freq_axis, 
            max_peaks, rfi_mask, true_params)


def plot_rfi_comparison(results_no_rfi, results_with_rfi, data_2d, time_axis, 
                        freq_axis, max_peaks, rfi_mask, true_params):
    """Plot comparison of results with and without RFI mitigation."""
    
    model_name = "exponential_2d"
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. Original 2D data with RFI
    ax = axes[0, 0]
    im = ax.imshow(data_2d, aspect='auto', origin='lower', 
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Data with RFI')
    
    # Mark RFI channels
    for i, is_rfi in enumerate(rfi_mask):
        if is_rfi:
            ax.axhline(freq_axis[i], color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.colorbar(im, ax=ax)
    
    # 2. Best-fit model WITHOUT RFI mitigation
    ax = axes[0, 1]
    best_idx_no_rfi = np.argmax(results_no_rfi.loglikelihood)
    best_params_no_rfi = results_no_rfi.particles[best_idx_no_rfi]
    
    from frbayes_jax.models import get_model_function
    model_func = get_model_function(model_name)
    model_2d_no_rfi = model_func(
        jnp.array(time_axis),
        jnp.array(freq_axis),
        best_params_no_rfi,
        max_peaks,
        fit_pulses=True
    )
    
    im = ax.imshow(np.array(model_2d_no_rfi), aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Best-fit (No RFI mitigation)')
    plt.colorbar(im, ax=ax)
    
    # 3. Best-fit model WITH RFI mitigation
    ax = axes[0, 2]
    best_idx_with_rfi = np.argmax(results_with_rfi.loglikelihood)
    best_params_with_rfi = results_with_rfi.particles[best_idx_with_rfi]
    
    # Remove log_p parameter for model evaluation
    model_params_with_rfi = best_params_with_rfi[:-1]
    
    model_2d_with_rfi = model_func(
        jnp.array(time_axis),
        jnp.array(freq_axis),
        model_params_with_rfi,
        max_peaks,
        fit_pulses=True
    )
    
    im = ax.imshow(np.array(model_2d_with_rfi), aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Best-fit (With RFI mitigation)')
    plt.colorbar(im, ax=ax)
    
    # 4. Residuals WITHOUT RFI mitigation
    ax = axes[1, 0]
    residuals_no_rfi = data_2d - np.array(model_2d_no_rfi)
    im = ax.imshow(residuals_no_rfi, aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                   cmap='RdBu_r', vmin=-10, vmax=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Residuals (No RFI mitigation)')
    plt.colorbar(im, ax=ax)
    
    # 5. Residuals WITH RFI mitigation
    ax = axes[1, 1]
    residuals_with_rfi = data_2d - np.array(model_2d_with_rfi)
    im = ax.imshow(residuals_with_rfi, aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                   cmap='RdBu_r', vmin=-10, vmax=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Residuals (With RFI mitigation)')
    plt.colorbar(im, ax=ax)
    
    # 6. Anomaly detection visualization
    ax = axes[1, 2]
    # Calculate which points are likely anomalies based on residuals
    log_p = best_params_with_rfi[-1]
    p = np.exp(log_p)
    threshold = np.max(np.abs(data_2d)) * p  # Simplified threshold
    
    anomaly_map = np.abs(residuals_with_rfi) > threshold
    im = ax.imshow(anomaly_map.astype(float), aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                   cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title(f'Detected Anomalies (p={p:.3f})')
    plt.colorbar(im, ax=ax)
    
    # 7. Spectral index posterior comparison
    ax = axes[2, 0]
    alpha_idx = get_spectral_index_location(model_name, max_peaks)
    if alpha_idx is not None:
        # No RFI mitigation
        param_names_no_rfi = get_param_names(model_name, max_peaks, fit_pulses=True)
        nested_samples_no_rfi = anesthetic.NestedSamples(
            data=results_no_rfi.particles,
            logL=results_no_rfi.loglikelihood,
            logL_birth=results_no_rfi.loglikelihood_birth,
            columns=param_names_no_rfi
        )
        alpha_samples_no_rfi = nested_samples_no_rfi.iloc[:, alpha_idx]
        
        # With RFI mitigation
        param_names_with_rfi = param_names_no_rfi + ['log_p']
        nested_samples_with_rfi = anesthetic.NestedSamples(
            data=results_with_rfi.particles,
            logL=results_with_rfi.loglikelihood,
            logL_birth=results_with_rfi.loglikelihood_birth,
            columns=param_names_with_rfi
        )
        alpha_samples_with_rfi = nested_samples_with_rfi.iloc[:, alpha_idx]
        
        ax.hist(alpha_samples_no_rfi, bins=30, alpha=0.5, color='red', 
                label=f'No RFI mit.: {alpha_samples_no_rfi.mean():.2f}±{alpha_samples_no_rfi.std():.2f}')
        ax.hist(alpha_samples_with_rfi, bins=30, alpha=0.5, color='blue',
                label=f'With RFI mit.: {alpha_samples_with_rfi.mean():.2f}±{alpha_samples_with_rfi.std():.2f}')
        ax.axvline(-1.4, color='green', linestyle='--', linewidth=2, label='True value: -1.4')
        ax.set_xlabel('Spectral Index α')
        ax.set_ylabel('Posterior Samples')
        ax.set_title('Spectral Index Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 8. Npulse posterior comparison
    ax = axes[2, 1]
    npulse_idx = get_npulse_index(model_name, max_peaks, fit_pulses=True)
    if npulse_idx is not None:
        npulse_samples_no_rfi = results_no_rfi.particles[:, npulse_idx]
        npulse_samples_with_rfi = results_with_rfi.particles[:, npulse_idx]
        
        bins = np.arange(0.5, max_peaks + 1.5, 1)
        ax.hist(npulse_samples_no_rfi, bins=bins, alpha=0.5, color='red', density=True,
                label=f'No RFI mit.: {np.mean(npulse_samples_no_rfi):.1f}')
        ax.hist(npulse_samples_with_rfi, bins=bins, alpha=0.5, color='blue', density=True,
                label=f'With RFI mit.: {np.mean(npulse_samples_with_rfi):.1f}')
        ax.axvline(3, color='green', linestyle='--', linewidth=2, label='True value: 3')
        ax.set_xlabel('Number of Pulses')
        ax.set_ylabel('Probability')
        ax.set_title('Npulse Comparison')
        ax.set_xticks(range(1, max_peaks + 1))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 9. Frequency-averaged profiles
    ax = axes[2, 2]
    profile_data = np.mean(data_2d, axis=0)
    profile_model_no_rfi = np.mean(model_2d_no_rfi, axis=0)
    profile_model_with_rfi = np.mean(model_2d_with_rfi, axis=0)
    
    ax.plot(time_axis, profile_data, 'k-', alpha=0.5, label='Data')
    ax.plot(time_axis, profile_model_no_rfi, 'r-', alpha=0.7, label='No RFI mit.')
    ax.plot(time_axis, profile_model_with_rfi, 'b-', alpha=0.7, label='With RFI mit.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity')
    ax.set_title('Frequency-Averaged Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('RFI Mitigation Comparison using Bayesian Anomaly Detection', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'results/rfi_mitigation_comparison_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: results/rfi_mitigation_comparison_{timestamp}.png")
    
    plt.show()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run test
    results = test_rfi_mitigation()
    
    if results is not None:
        (results_no_rfi, results_with_rfi, data_2d, time_axis, freq_axis, 
         max_peaks, rfi_mask, true_params) = results
        
        # Plot comparison
        plot_rfi_comparison(results_no_rfi, results_with_rfi, data_2d, time_axis, 
                           freq_axis, max_peaks, rfi_mask, true_params)
    
    print("\nDone!")