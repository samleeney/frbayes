#!/usr/bin/env python3
"""
Complete analysis pipeline: combines analyze_frb_paper_exact.py and run_analysis_clean.py
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

# ============================================================
# PART 1: RUN analyze_frb_paper_exact.py
# ============================================================

# Configuration
model_name = 'exponential_with_baseline'
min_peaks = 1
max_peaks = 12
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


# Plot the preprocessed data
plt.figure(figsize=(12, 4))
plt.plot(t, profile, 'k-', alpha=0.7, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('S/N')
plt.title('FRB 20191221A - Paper Preprocessing (S/N)')
plt.grid(True, alpha=0.3)
plt.savefig('frb_paper_preprocessed_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved preprocessed profile")

# Calculate dimensions
ndims = 3 * max_peaks + 1  # A, tau, u for each peak + sigma
ndims += 1  # baseline
if fit_pulses:
    ndims += 1  # Npulse

num_live_points = ndims * 25

print(f"Running nested sampling ({num_live_points} live points, {min_peaks}-{max_peaks} pulses)...")

# Custom prior bounds to reduce baseline degeneracy
prior_bounds = {
    'amplitude': {'min': 0, 'max': 50},
    'tau': {'min': 0.001, 'max': 0.5},
    'u': {'min': 0, 'max': 1},
    'baseline': {'min': -0.5, 'max': 0},  # Negative only to avoid degeneracy
    'log_sigma': {'min': np.log(0.1), 'max': np.log(2.0)},
    'npulse': {'min': min_peaks, 'max': max_peaks}
}

# Nested sampling parameters
ns_params = {
    'num_live_points': num_live_points,
    'num_delete': num_live_points // 2,
    'num_inner_steps': ndims * 5,
    'log_tolerance': -3.0,
    'seed': 42
}

# Normalize time for model
t_normalized = (t - t[0]) / (t[-1] - t[0])

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

print("Nested sampling completed")

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
chains_file = os.path.join(chains_dir, f'nested_samples_{min_peaks}to{max_peaks}peaks.csv')
nested_samples.to_csv(chains_file)
print("Saved chains to CSV")

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

print(f"Fitted {fitted_npulse_rounded} pulses (log evidence: {np.max(logL_valid):.2f})")

# Save results
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

np.savez(f'frb_paper_results_{min_peaks}to{max_peaks}peaks.npz', **results_dict)

# ============================================================
# PART 2: RUN run_analysis_clean.py
# ============================================================


# Load chains using anesthetic.read_csv
samples = read_csv(chains_file)

output_dir = 'analysis_results'
os.makedirs(output_dir, exist_ok=True)

# 1. Corner plot
print("Creating corner plot...")

# Select parameters to plot
params_to_plot = []
for i in range(min(fitted_npulse_rounded, 3)):
    params_to_plot.extend([
        f'A__{i+1}',
        f'tau__{i+1}',
        f'u__{i+1}'
    ])
params_to_plot.extend(['B__text_offset', 'sigma', 'N__text_pulse'])

# Filter to existing columns
params_to_plot = [p for p in params_to_plot if p in samples.columns][:12]

fig, axes = make_2d_axes(params_to_plot, figsize=(12, 12))
samples.plot_2d(axes, kinds='kde', alpha=0.8, color='green')

fig.suptitle('FRB 20191221A - Parameter Correlations', fontsize=14)
fig.tight_layout()

output_file = os.path.join(output_dir, "corner_plot.png")
fig.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

# 2. Npulse distribution
print("Creating Npulse distribution...")
fig, ax = plt.subplots(figsize=(8, 6))

npulse_data = samples['N__text_pulse']
weights = samples.get_weights()

counts, bins, _ = ax.hist(npulse_data, bins=np.arange(0.5, 13.5, 1),
                          weights=weights,
                          color='purple', alpha=0.7,
                          edgecolor='black', linewidth=1.5)

ax.set_xlabel('Number of Pulses', fontsize=13)
ax.set_ylabel('Posterior Probability', fontsize=13)
ax.set_title('FRB 20191221A - Pulse Number Distribution', fontsize=15)
ax.set_xticks(range(1, 13))
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
print("Creating functional posterior...")

model_func = get_model_function(model_name)

# Define wrapper for fgivenx
def model_wrapper(t_val, theta):
    """Wrapper to make model compatible with fgivenx."""
    # Normalize time
    t_norm = (np.array(t_val) - t[0]) / (t[-1] - t[0])
    t_array = jnp.array(t_norm)
    theta_array = jnp.array(theta)
    return np.array(model_func(t_array, theta_array, max_peaks, fit_pulses=True))

# Sample subset for speed
nsamples = 500
if len(samples) > nsamples:
    indices = np.random.choice(len(samples), nsamples, replace=False)
    samples_subset = samples.iloc[indices]
else:
    samples_subset = samples

# Get parameter columns only
param_cols = [col for col in samples_subset.columns
              if col not in ['weights', 'logL', 'logL_birth', 'nlive'] and not col.startswith('Unnamed')]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot contours
plot_contours(
    model_wrapper,
    t,
    samples_subset[param_cols],
    axes[0],
    weights=samples_subset.get_weights(),
    colors=plt.cm.Greens_r,
    cache=f"{output_dir}/cache_contours"
)

# Overlay data
axes[0].plot(t, profile, 'k.', alpha=0.5, markersize=2, label='Data')
axes[0].legend()
axes[0].set_ylabel('S/N', fontsize=12)
axes[0].set_title('Functional Posterior', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot lines
plot_lines(
    model_wrapper,
    t,
    samples_subset[param_cols],
    axes[1],
    weights=samples_subset.get_weights(),
    color='green',
    alpha=0.1,
    cache=f"{output_dir}/cache_lines"
)

# Overlay data
axes[1].plot(t, profile, 'k.', alpha=0.5, markersize=2, label='Data')
axes[1].legend()
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('S/N', fontsize=12)
axes[1].set_title('Model Realizations', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(output_dir, "functional_posterior.png")
fig.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"Analysis complete. Plots saved to {output_dir}/")