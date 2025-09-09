#!/usr/bin/env python3
"""
2-peak test with FIXED number of pulses (fit_pulses=False)
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

# 2 peak model with FIXED number of peaks (9 dimensions: 2×4 params + sigma)
model = get_model('emg', ModelConfig(max_peaks=2, fit_pulses=False, model_type='emg'))

# Generate data with 2 peaks
t = jnp.linspace(0, 5, 100)
true_params = {
    'A': jnp.array([0.08, 0.06]),      # 2 amplitudes
    'tau': jnp.array([0.5, 0.3]),      # 2 scattering timescales
    'u': jnp.array([1.5, 3.0]),        # 2 arrival times
    'w': jnp.array([0.2, 0.15]),       # 2 widths
    'sigma': jnp.array([0.003])        # noise
}

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
        'sigma': {'min': 1e-4, 'max': 0.01}
    },
    config={'num_live_points': nlive, 'max_samples': 50000, 
            'precision_criterion': 3.0, 'num_delete': ndelete, 'num_inner_steps': 5}
)

print(f"Running 2-peak FIXED test (ndims={ndims})...")
results = sampler.run(key, nlive, 50000, 3.0)

# Output directory
output_dir = Path(__file__).parent / "2peak_fixed_output"
output_dir.mkdir(exist_ok=True, parents=True)

# Get final_state from results
final_state = results['final_info']

# Create NestedSamples object exactly as in documentation
nested_samples = anesthetic.NestedSamples(
    data=final_state.particles,
    logL=final_state.loglikelihood,
    logL_birth=final_state.loglikelihood_birth,
    columns=['A1', 'A2', 'tau1', 'tau2', 'u1', 'u2', 'w1', 'w2', 'sigma']
)
ns = nested_samples  # Keep ns variable for compatibility

# Create corner plot
print("\nCreating corner plot...")
axes = ns[['A1', 'A2', 'u1', 'u2', 'sigma']].plot_2d()
plt.gcf().savefig(output_dir / "corner_plot.png", dpi=150)
plt.close()
print("Corner plot saved successfully")

# Time array for plotting (reduced to 30 points for speed)
t_plot = np.linspace(0, 5, 30)

# Define model function
def model_fn(params_array, times):
    """Convert flat parameter array to model prediction."""
    params = {
        'A': jnp.array(params_array[0:2]),
        'tau': jnp.array(params_array[2:4]),
        'u': jnp.array(params_array[4:6]),
        'w': jnp.array(params_array[6:8]),
        'sigma': jnp.array([params_array[8]])
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
    true_params['sigma']
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
ax.set_title('Functional Posterior (2 peaks, FIXED)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

fig.savefig(output_dir / 'functional_posterior.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved to {output_dir}/")