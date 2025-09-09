#!/usr/bin/env python3
"""
2-peak test with VARIABLE number of pulses (fit_pulses=True)
Tests model selection between 1-3 peaks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import anesthetic
import fgivenx

from models import get_model, ModelConfig
from inference.blackjax_nested_sampling import BlackJAXNestedSampler
from utils import configure_gpu

configure_gpu(memory_fraction=0.8, force_cpu=False)

# Model with up to 3 peaks, fitting number of pulses
model = get_model('emg', ModelConfig(max_peaks=3, fit_pulses=True, model_type='emg'))

# Generate data with 2 peaks (same as fixed test)
t = jnp.linspace(0, 5, 100)
true_params = {
    'A': jnp.array([0.08, 0.06, 0.0]),  # 2 peaks, 3rd amplitude is 0
    'tau': jnp.array([0.5, 0.3, 0.1]),   # scattering timescales
    'u': jnp.array([1.5, 3.0, 4.5]),     # arrival times
    'w': jnp.array([0.2, 0.15, 0.1]),    # widths
    'sigma': jnp.array([0.003]),         # noise
    'Npulse': jnp.array([2.0])           # True number of pulses = 2
}

# Generate noisy data
key = jax.random.PRNGKey(42)
pp = model.model_function(t, true_params)
pp += true_params['sigma'][0] * jax.random.normal(jax.random.split(key)[1], pp.shape)
data = {'pp': pp, 't': t}

# Run nested sampling
ndims = model.ndims
nlive = ndims * 25
ndelete = nlive // 2

sampler = BlackJAXNestedSampler(
    model, data,
    prior_config={
        'amplitude': {'min': 0.001, 'max': 0.2},
        'tau': {'min': 0.1, 'max': 1.0},
        'u': {'min': 0.0, 'max': 5.0},
        'width': {'min': 0.05, 'max': 0.5},
        'sigma': {'min': 1e-4, 'max': 0.01},
        'Npulse': {'min': 1, 'max': 3}  # Prior on number of pulses
    },
    config={'num_live_points': nlive, 'max_samples': 50000, 
            'precision_criterion': 3.0, 'num_delete': ndelete, 'num_inner_steps': 5}
)

print(f"Running 2-peak FIT_PULSES test (ndims={ndims}, Npulse prior=[1,3])...")
results = sampler.run(key, nlive, 50000, 3.0)

# Output directory
output_dir = Path(__file__).parent / "2peak_fitpulses_output"
output_dir.mkdir(exist_ok=True, parents=True)

# Get final_state from results
final_state = results['final_info']

# Create NestedSamples object exactly as in documentation
nested_samples = anesthetic.NestedSamples(
    data=final_state.particles,
    logL=final_state.loglikelihood,
    logL_birth=final_state.loglikelihood_birth,
    columns=['A1', 'A2', 'A3', 'tau1', 'tau2', 'tau3', 
             'u1', 'u2', 'u3', 'w1', 'w2', 'w3', 'sigma', 'Npulse']
)
ns = nested_samples  # Keep ns variable for compatibility

# Create corner plot
print("\nCreating corner plot...")
axes = ns[['Npulse', 'A1', 'A2', 'u1', 'u2', 'sigma']].plot_2d()
plt.gcf().savefig(output_dir / "corner_plot.png", dpi=150)
plt.close()
print("Corner plot saved successfully")

# Create 1D histogram of Npulse
fig, ax = plt.subplots(figsize=(8, 6))
ns['Npulse'].plot.hist(bins=30, density=True, ax=ax, alpha=0.7, color='blue')
ax.axvline(2.0, color='red', linestyle='--', label='True value', linewidth=2)
ax.set_xlabel('$N_{\\mathrm{pulse}}$')
ax.set_ylabel('Posterior probability')
ax.set_title('Posterior on Number of Pulses')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(output_dir / "npulse_posterior.png", dpi=150)
plt.close()

# Time array for plotting (reduced for speed)
t_plot = np.linspace(0, 5, 30)

# Define model function
def model_fn(params_array, times):
    """Convert flat parameter array to model prediction."""
    params = {
        'A': jnp.array(params_array[0:3]),
        'tau': jnp.array(params_array[3:6]),
        'u': jnp.array(params_array[6:9]),
        'w': jnp.array(params_array[9:12]),
        'sigma': jnp.array([params_array[12]]),
        'Npulse': jnp.array([params_array[13]])
    }
    return np.array(model.model_function(times, params))

# Define function for fgivenx
def f(x, params):
    """Function for fgivenx that returns model values."""
    return model_fn(params, x)

# Get samples as numpy array
samples = np.array(results['particles_dead'])

# Convert true parameters to array
true_params_array = np.concatenate([
    true_params['A'],
    true_params['tau'],
    true_params['u'],
    true_params['w'],
    true_params['sigma'],
    true_params['Npulse']
])

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot contours using fgivenx
try:
    _ = fgivenx.plot_contours(f, t_plot, samples, ax, colors=plt.cm.Blues_r)
except AttributeError:
    pass

# Plot truth
truth_prediction = f(t_plot, true_params_array)
ax.plot(t_plot, truth_prediction, 'r-', label='Truth (2 peaks)', linewidth=2, alpha=0.8)

# Plot data
ax.scatter(np.array(data['t']), np.array(data['pp']), 
          c='black', s=5, alpha=0.5, label='Data', zorder=10)

ax.set_xlabel('Time')
ax.set_ylabel('Signal')
ax.set_title('Functional Posterior (fit_pulses=True)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

fig.savefig(output_dir / 'functional_posterior.png', dpi=150, bbox_inches='tight')
plt.close()

# Get Npulse samples and check model selection
npulse_samples = samples[:, 13]
n2_frac = np.mean((npulse_samples >= 1.5) & (npulse_samples < 2.5))
print(f"Fraction favoring 2 peaks: {n2_frac:.1%}")
print(f"Saved to {output_dir}/")