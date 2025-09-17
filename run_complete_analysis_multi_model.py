#!/usr/bin/env python3
"""
Complete analysis pipeline for multiple models: EMG and periodic_exponential
"""
import numpy as np
import matplotlib.pyplot as plt
from anesthetic import NestedSamples, read_csv, make_2d_axes
from fgivenx import plot_contours, plot_lines
import jax.numpy as jnp
import sys
import os

# Add frbayes_jax to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frbayes_jax'))

from frbayes_jax.data import preprocess_data
from frbayes_jax.sampling import run_nested_sampling
from frbayes_jax.models import get_model_function, get_param_names

# Configuration
models_to_run = ['exponential_with_baseline', 'emg', 'periodic_exponential']  # Run all three models
min_peaks = 1
max_peaks = 15
fit_pulses = True

# Data file paths
data_file = '/home/sam/frbayes_fresh/data_frbayes/20191221A_original_data.h5'

# Paper's exact preprocessing parameters
original_time_res = 0.98304e-3
original_freq_res = 0.390625e6
desired_time_res = 7.86432e-3
desired_freq_res = 3.125e6
freq_min = 400.0
freq_max = 800.0

print(f"Loading and preprocessing data...")

# Use the paper preprocessing function
wfall_downsampled, pulse_profile_snr, time_axis = preprocess_data(
    data_file=data_file,
    original_freq_res=original_freq_res,
    original_time_res=original_time_res,
    desired_freq_res=desired_freq_res,
    desired_time_res=desired_time_res,
    freq_min=freq_min,
    freq_max=freq_max,
    preprocessing_mode="paper"
)

profile = pulse_profile_snr
t = time_axis

# Further downsample if needed
target_points = 500
if len(profile) > target_points:
    downsample_factor = len(profile) // target_points
    profile = profile[::downsample_factor]
    t = t[::downsample_factor]

# Plot the preprocessed data once
plt.figure(figsize=(12, 4))
plt.plot(t, profile, 'k-', alpha=0.7, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('S/N')
plt.title('FRB 20191221A - Paper Preprocessing (S/N)')
plt.grid(True, alpha=0.3)
plt.savefig('frb_paper_preprocessed_profile_multi_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved preprocessed profile")

# Normalize time for model
t_normalized = (t - t[0]) / (t[-1] - t[0])

# Store results for comparison
model_results = {}

# ============================================================
# RUN ANALYSIS FOR EACH MODEL
# ============================================================

for model_name in models_to_run:
    print(f"\n{'='*60}")
    print(f"Running analysis for model: {model_name}")
    print(f"{'='*60}\n")

    # Calculate dimensions based on model
    if model_name == 'periodic_exponential':
        # For periodic model: period, phase, A_template, tau, sigma, baseline
        ndims = 6
        if fit_pulses:
            ndims += 1  # Npulse
    else:
        # For EMG and other models
        ndims = 3 * max_peaks + 1  # A, tau, u for each peak + sigma
        ndims += 1  # baseline
        if fit_pulses:
            ndims += 1  # Npulse

    num_live_points = ndims * 25  # Standard recommendation for memory efficiency

    print(f"Running nested sampling ({num_live_points} live points, model={model_name})...")

    # Custom prior bounds based on model
    if model_name == 'periodic_exponential':
        prior_bounds = {
            'period': {'min': 0.1, 'max': 0.5},  # Period in normalized time
            'phase': {'min': 0, 'max': 1},
            'amplitude': {'min': 0, 'max': 50},
            'tau': {'min': 0.001, 'max': 0.1},
            'baseline': {'min': -1.0, 'max': 1.0},
            'log_sigma': {'min': np.log(0.1), 'max': np.log(2.0)},
            'npulse': {'min': min_peaks, 'max': max_peaks}
        }
    else:
        prior_bounds = {
            'amplitude': {'min': 0, 'max': 50},
            'tau': {'min': 0.001, 'max': 0.5},
            'u': {'min': 0, 'max': 1},
            'baseline': {'min': -1.0, 'max': 1.0},
            'log_sigma': {'min': np.log(0.1), 'max': np.log(2.0)},
            'npulse': {'min': min_peaks, 'max': max_peaks}
        }

    # Nested sampling parameters
    ns_params = {
        'num_live_points': num_live_points,
        'num_delete': num_live_points // 2,
        'num_inner_steps': ndims * 20,  # Doubled for better MCMC mixing
        'log_tolerance': -3.0,
        'seed': 42
    }

    # Run nested sampling
    final_state = run_nested_sampling(
        model_name=model_name,
        data=profile,
        t=t_normalized,
        prior_bounds=prior_bounds,
        max_peaks=max_peaks,
        fit_pulses=fit_pulses,
        **ns_params
    )

    print(f"Nested sampling completed for {model_name}")

    # Get parameter names
    param_names_latex = get_param_names(model_name, max_peaks, fit_pulses)
    param_names = []
    for name in param_names_latex:
        clean_name = name.replace('$', '').replace('\\', '')
        clean_name = clean_name.replace('{', '_').replace('}', '')
        param_names.append(clean_name)

    # Create NestedSamples object
    nested_samples = NestedSamples(
        data=final_state.particles,
        columns=param_names,
        logL=final_state.loglikelihood,
        logL_birth=final_state.loglikelihood_birth if hasattr(final_state, 'loglikelihood_birth') else None
    )

    # Save the NestedSamples object as CSV
    chains_dir = 'chains'
    os.makedirs(chains_dir, exist_ok=True)
    chains_file = os.path.join(chains_dir, f'nested_samples_{model_name}_{min_peaks}to{max_peaks}peaks.csv')
    nested_samples.to_csv(chains_file)
    print(f"Saved chains to {chains_file}")

    # Process results
    logL = final_state.loglikelihood
    valid_mask = ~np.isnan(logL) & ~np.isinf(logL)
    logL_valid = logL[valid_mask]
    particles_valid = final_state.particles[valid_mask]

    # Filter for valid Npulse range
    if fit_pulses:
        npulse_values = particles_valid[:, -1]
        pulse_mask = (npulse_values >= min_peaks) & (npulse_values <= max_peaks)
        if np.any(pulse_mask):
            logL_valid = logL_valid[pulse_mask]
            particles_valid = particles_valid[pulse_mask]

    # Get best fit
    best_idx = np.argmax(logL_valid)
    best_fit = particles_valid[best_idx]
    fitted_npulse = best_fit[-1] if fit_pulses else max_peaks
    fitted_npulse_rounded = int(np.round(fitted_npulse))

    print(f"Model {model_name}: Fitted {fitted_npulse_rounded} pulses (log evidence: {np.max(logL_valid):.2f})")

    # Save model-specific results
    results_dict = {
        'best_fit': best_fit,
        'fitted_npulse': fitted_npulse_rounded,
        't': t,
        'data': profile,
        'log_evidence': np.max(logL_valid),
        'particles': particles_valid,
        'max_peaks': max_peaks,
        'model_name': model_name
    }

    np.savez(f'frb_paper_results_{model_name}_{min_peaks}to{max_peaks}peaks.npz', **results_dict)

    # Store for comparison
    model_results[model_name] = {
        'nested_samples': nested_samples,
        'chains_file': chains_file,
        'best_fit': best_fit,
        'fitted_npulse': fitted_npulse_rounded,
        'log_evidence': np.max(logL_valid),
        'param_names': param_names
    }

    # ============================================================
    # CREATE MODEL-SPECIFIC PLOTS
    # ============================================================

    output_dir = f'analysis_results_{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Corner plot
    print(f"Creating corner plot for {model_name}...")

    # Select parameters to plot based on model
    if model_name == 'periodic_exponential':
        params_to_plot = ['P', 'phi', 'A__template', 'tau', 'B__text_offset', 'sigma']
        if fit_pulses:
            params_to_plot.append('N__text_pulse')
    else:
        params_to_plot = []
        for i in range(min(fitted_npulse_rounded, 3)):
            params_to_plot.extend([f'A__{i+1}', f'tau__{i+1}'])
            if model_name == 'emg':
                params_to_plot.append(f'u__{i+1}')
            else:  # exponential_with_baseline uses t parameters
                params_to_plot.append(f't__{i+1}')
        params_to_plot.extend(['B__text_offset', 'sigma', 'N__text_pulse'])

    # Filter to existing columns
    params_to_plot = [p for p in params_to_plot if p in nested_samples.columns][:12]

    fig, axes = make_2d_axes(params_to_plot, figsize=(12, 12))
    if model_name == 'emg':
        plot_color = 'blue'
    elif model_name == 'exponential_with_baseline':
        plot_color = 'green'
    else:  # periodic_exponential
        plot_color = 'red'
    nested_samples.plot_2d(axes, kinds='kde', alpha=0.8, color=plot_color)

    fig.suptitle(f'FRB 20191221A - {model_name.upper()} Model - Parameter Correlations', fontsize=14)
    fig.tight_layout()

    output_file = os.path.join(output_dir, "corner_plot.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Npulse distribution (if fitting pulses)
    if fit_pulses:
        print(f"Creating Npulse distribution for {model_name}...")
        fig, ax = plt.subplots(figsize=(8, 6))

        npulse_data = nested_samples['N__text_pulse']
        weights = nested_samples.get_weights()

        if model_name == 'emg':
            hist_color = 'blue'
        elif model_name == 'exponential_with_baseline':
            hist_color = 'green'
        else:  # periodic_exponential
            hist_color = 'red'

        counts, bins, _ = ax.hist(npulse_data, bins=np.arange(0.5, max_peaks+1.5, 1),
                                  weights=weights,
                                  color=hist_color,
                                  alpha=0.7,
                                  edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Number of Pulses', fontsize=13)
        ax.set_ylabel('Posterior Probability', fontsize=13)
        ax.set_title(f'FRB 20191221A - {model_name.upper()} - Pulse Number Distribution', fontsize=15)
        ax.set_xticks(range(1, max_peaks+1))
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        mean_npulse = np.average(npulse_data, weights=weights)
        std_npulse = np.sqrt(np.average((npulse_data - mean_npulse)**2, weights=weights))

        mode_idx = np.argmax(counts)
        mode_npulse = int((bins[mode_idx] + bins[mode_idx + 1]) / 2)

        ax.axvline(mean_npulse, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_npulse:.2f} ± {std_npulse:.2f}')
        ax.axvline(mode_npulse, color='green', linestyle='--', linewidth=2,
                  label=f'Mode: {mode_npulse}')

        ax.legend(fontsize=12, loc='upper right')

        output_file = os.path.join(output_dir, "npulse_distribution.png")
        fig.tight_layout()
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 3. Functional posterior using fgivenx
    print(f"Creating functional posterior for {model_name}...")

    model_func = get_model_function(model_name)

    # Define wrapper for fgivenx
    def model_wrapper(t_val, theta):
        """Wrapper to make model compatible with fgivenx."""
        # Normalize time
        t_norm = (np.array(t_val) - t[0]) / (t[-1] - t[0])
        t_array = jnp.array(t_norm)
        theta_array = jnp.array(theta)
        return np.array(model_func(t_array, theta_array, max_peaks, fit_pulses=fit_pulses))

    # Sample subset for speed
    nsamples = 500
    if len(nested_samples) > nsamples:
        indices = np.random.choice(len(nested_samples), nsamples, replace=False)
        samples_subset = nested_samples.iloc[indices]
    else:
        samples_subset = nested_samples

    # Get parameter columns only
    param_cols = [col for col in samples_subset.columns
                  if col not in ['weights', 'logL', 'logL_birth', 'nlive'] and not col.startswith('Unnamed')]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot contours
    if model_name == 'emg':
        cmap = plt.cm.Blues_r
    elif model_name == 'exponential_with_baseline':
        cmap = plt.cm.Greens_r
    else:  # periodic_exponential
        cmap = plt.cm.Reds_r
    plot_contours(
        model_wrapper,
        t,
        samples_subset[param_cols],
        axes[0],
        weights=samples_subset.get_weights(),
        colors=cmap,
        cache=f"{output_dir}/cache_contours_{model_name}"
    )

    # Overlay data
    axes[0].plot(t, profile, 'k.', alpha=0.5, markersize=2, label='Data')
    axes[0].legend()
    axes[0].set_ylabel('S/N', fontsize=12)
    axes[0].set_title(f'Functional Posterior - {model_name.upper()}', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot lines
    if model_name == 'emg':
        line_color = 'blue'
    elif model_name == 'exponential_with_baseline':
        line_color = 'green'
    else:  # periodic_exponential
        line_color = 'red'
    plot_lines(
        model_wrapper,
        t,
        samples_subset[param_cols],
        axes[1],
        weights=samples_subset.get_weights(),
        color=line_color,
        alpha=0.1,
        cache=f"{output_dir}/cache_lines_{model_name}"
    )

    # Overlay data
    axes[1].plot(t, profile, 'k.', alpha=0.5, markersize=2, label='Data')
    axes[1].legend()
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('S/N', fontsize=12)
    axes[1].set_title(f'Model Realizations - {model_name.upper()}', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "functional_posterior.png")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Analysis complete for {model_name}. Plots saved to {output_dir}/")

# ============================================================
# CREATE MODEL COMPARISON PLOTS
# ============================================================

print(f"\n{'='*60}")
print("Creating model comparison plots...")
print(f"{'='*60}\n")

comparison_dir = 'analysis_results_comparison'
os.makedirs(comparison_dir, exist_ok=True)

# Compare log evidences
fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(model_results.keys())
log_evidences = [model_results[m]['log_evidence'] for m in model_names]
fitted_npulses = [model_results[m]['fitted_npulse'] for m in model_names]

x_pos = np.arange(len(model_names))
colors = []
for m in model_names:
    if m == 'emg':
        colors.append('blue')
    elif m == 'exponential_with_baseline':
        colors.append('green')
    else:  # periodic_exponential
        colors.append('red')

bars = ax.bar(x_pos, log_evidences, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, val, npulse) in enumerate(zip(bars, log_evidences, fitted_npulses)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}\n({npulse} pulses)',
            ha='center', va='bottom', fontsize=11)

ax.set_xlabel('Model', fontsize=13)
ax.set_ylabel('Log Evidence', fontsize=13)
ax.set_title('Model Comparison: Log Evidence', fontsize=15)
ax.set_xticks(x_pos)
ax.set_xticklabels([m.upper() for m in model_names])
ax.grid(True, alpha=0.3, axis='y')

# Calculate Bayes factors for best model
if len(log_evidences) > 1:
    best_idx = np.argmax(log_evidences)
    best_model = model_names[best_idx]
    best_log_ev = log_evidences[best_idx]

    bf_text = f"Best Model: {best_model.upper()}\n"
    for i, (m, le) in enumerate(zip(model_names, log_evidences)):
        if i != best_idx:
            bf = np.exp(best_log_ev - le)
            bf_text += f"BF({best_model}/{m}): {bf:.2e}\n"

    ax.text(0.95, 0.95, bf_text.strip(),
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'model_comparison_log_evidence.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save comparison summary
summary_file = os.path.join(comparison_dir, 'model_comparison_summary.txt')
with open(summary_file, 'w') as f:
    f.write("FRB 20191221A - Model Comparison Summary\n")
    f.write("=" * 50 + "\n\n")

    for model_name in model_names:
        f.write(f"Model: {model_name.upper()}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Log Evidence: {model_results[model_name]['log_evidence']:.2f}\n")
        f.write(f"Fitted Npulse: {model_results[model_name]['fitted_npulse']}\n")
        f.write("\n")

    if len(model_names) > 1:
        f.write("=" * 50 + "\n")
        f.write("Bayes Factors:\n")
        best_idx = np.argmax(log_evidences)
        best_model = model_names[best_idx]
        best_log_ev = log_evidences[best_idx]
        f.write(f"Best Model: {best_model.upper()} (Log Z = {best_log_ev:.2f})\n\n")

        for i, (m, le) in enumerate(zip(model_names, log_evidences)):
            if i != best_idx:
                bf = np.exp(best_log_ev - le)
                f.write(f"BF({best_model}/{m}): {bf:.2e}\n")

        f.write(f"\nEvidence favors {best_model.upper()} model\n")

print(f"Model comparison complete. Results saved to {comparison_dir}/")
print(f"Summary saved to {summary_file}")

# Final summary
print(f"\n{'='*60}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*60}")
for model_name in model_names:
    print(f"{model_name.upper():20} | Log Z: {model_results[model_name]['log_evidence']:8.2f} | Npulse: {model_results[model_name]['fitted_npulse']:2d}")
print(f"{'='*60}\n")