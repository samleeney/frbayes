#!/usr/bin/env python3
"""
Run complete analysis with fixed npulse values from 1 to 15
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
model_name = 'exponential_with_baseline'
fit_pulses = False  # Fixed npulse, not fitting

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
plt.savefig('frb_paper_preprocessed_profile_fixed_npulse.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved preprocessed profile")

# Normalize time for model
t_normalized = (t - t[0]) / (t[-1] - t[0])

# Store results for all npulse values
all_results = {}
all_log_evidences = []
npulse_values = list(range(1, 16))

# Run analysis for each fixed npulse value
for n_pulses in npulse_values:
    print(f"\n{'='*60}")
    print(f"Running analysis with fixed npulse = {n_pulses}")
    print(f"{'='*60}")

    # Calculate dimensions for this npulse value
    ndims = 3 * n_pulses + 1  # A, tau, u for each peak + sigma
    ndims += 1  # baseline
    # No npulse parameter since we're fixing it

    num_live_points = ndims * 25  # Standard recommendation

    print(f"Running nested sampling ({num_live_points} live points, {n_pulses} pulses fixed)...")

    # Custom prior bounds
    prior_bounds = {
        'amplitude': {'min': 0, 'max': 50},
        'tau': {'min': 0.001, 'max': 0.5},
        'u': {'min': 0, 'max': 1},
        'baseline': {'min': -1.0, 'max': 1.0},
        'log_sigma': {'min': np.log(0.1), 'max': np.log(2.0)}
    }

    # Nested sampling parameters
    ns_params = {
        'num_live_points': num_live_points,
        'num_delete': num_live_points // 2,
        'num_inner_steps': ndims * 20,
        'log_tolerance': -3.0,
        'seed': 42
    }

    try:
        # Run nested sampling with fixed npulse
        final_state = run_nested_sampling(
            model_name=model_name,
            data=profile,
            t=t_normalized,
            prior_bounds=prior_bounds,
            max_peaks=n_pulses,  # Set max_peaks to the fixed value
            fit_pulses=False,     # Don't fit npulse
            **ns_params
        )

        print(f"Nested sampling completed for npulse={n_pulses}")

        # Get parameter names
        param_names_latex = get_param_names(model_name, n_pulses, fit_pulses=False)
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
        chains_dir = 'chains_fixed_npulse'
        os.makedirs(chains_dir, exist_ok=True)
        chains_file = os.path.join(chains_dir, f'nested_samples_npulse{n_pulses:02d}.csv')
        nested_samples.to_csv(chains_file)
        print(f"Saved chains to {chains_file}")

        # Process results
        logL = final_state.loglikelihood
        valid_mask = ~np.isnan(logL) & ~np.isinf(logL)
        logL_valid = logL[valid_mask]
        particles_valid = final_state.particles[valid_mask]

        # Get best fit and log evidence
        best_idx = np.argmax(logL_valid)
        best_fit = particles_valid[best_idx]
        log_evidence = np.max(logL_valid)

        print(f"npulse={n_pulses}: log evidence = {log_evidence:.2f}")

        # Store results
        all_results[n_pulses] = {
            'best_fit': best_fit,
            'log_evidence': log_evidence,
            'particles': particles_valid,
            'logL': logL_valid,
            'nested_samples': nested_samples,
            'chains_file': chains_file
        }
        all_log_evidences.append(log_evidence)

        # Save individual results
        np.savez(f'frb_paper_results_npulse{n_pulses:02d}_fixed.npz',
                 best_fit=best_fit,
                 npulse=n_pulses,
                 t=t,
                 data=profile,
                 log_evidence=log_evidence,
                 particles=particles_valid,
                 model_name=model_name)

    except Exception as e:
        print(f"Error for npulse={n_pulses}: {e}")
        all_log_evidences.append(np.nan)
        continue

# Create summary plot of log evidence vs npulse
print("\n" + "="*60)
print("Creating summary plots...")
print("="*60)

output_dir = 'analysis_results_fixed_npulse'
os.makedirs(output_dir, exist_ok=True)

# Plot log evidence vs npulse
fig, ax = plt.subplots(figsize=(10, 6))
valid_npulse = [n for n, le in zip(npulse_values, all_log_evidences) if not np.isnan(le)]
valid_log_evidence = [le for le in all_log_evidences if not np.isnan(le)]

ax.plot(valid_npulse, valid_log_evidence, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Pulses', fontsize=12)
ax.set_ylabel('Log Evidence', fontsize=12)
ax.set_title('Model Comparison: Log Evidence vs Number of Pulses', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 16))

# Mark the maximum
if valid_log_evidence:
    max_idx = np.argmax(valid_log_evidence)
    best_npulse = valid_npulse[max_idx]
    best_log_evidence = valid_log_evidence[max_idx]
    ax.plot(best_npulse, best_log_evidence, 'r*', markersize=15,
            label=f'Best: npulse={best_npulse}, log Z={best_log_evidence:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'log_evidence_vs_npulse.png'), dpi=150, bbox_inches='tight')
plt.close()

# Calculate Bayes factors relative to best model
if valid_log_evidence:
    print(f"\nBayes Factors (relative to npulse={best_npulse}):")
    print("-" * 40)
    for n, le in zip(valid_npulse, valid_log_evidence):
        bayes_factor = np.exp(le - best_log_evidence)
        print(f"npulse={n:2d}: BF = {bayes_factor:.3e}")

# Save summary
summary_file = os.path.join(output_dir, 'model_comparison_summary.txt')
with open(summary_file, 'w') as f:
    f.write("FRB 20191221A - Fixed npulse Model Comparison\n")
    f.write("=" * 50 + "\n\n")
    f.write("npulse\tlog_evidence\tBayes_factor\n")
    f.write("-" * 40 + "\n")

    if valid_log_evidence:
        for n, le in zip(valid_npulse, valid_log_evidence):
            bf = np.exp(le - best_log_evidence)
            f.write(f"{n}\t{le:.2f}\t{bf:.3e}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Best model: npulse = {best_npulse}\n")
        f.write(f"Log evidence: {best_log_evidence:.2f}\n")

print(f"\nAnalysis complete. Results saved to {output_dir}/")
print(f"Summary saved to {summary_file}")

# Save all results in a single file
np.savez('all_fixed_npulse_results.npz',
         npulse_values=np.array(valid_npulse),
         log_evidences=np.array(valid_log_evidence),
         t=t,
         data=profile)

print("\nAll results saved to all_fixed_npulse_results.npz")