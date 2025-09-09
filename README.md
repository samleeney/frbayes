# FRBayes - JAX Implementation

A JAX-based implementation of FRBayes for Fast Radio Burst (FRB) parameter inference using nested sampling with BlackJAX.

## Features

- **JAX-based models**: Exponential Modified Gaussian (EMG), Exponential, and Periodic models
- **Model selection**: Variable number of pulses with continuous Npulse parameter
- **Nested sampling**: BlackJAX implementation with proper convergence criteria
- **GPU acceleration**: Automatic GPU detection and configuration
- **Visualization**: Corner plots and functional posteriors with anesthetic and fgivenx

## Installation

```bash
# Install dependencies
pip install jax jaxlib blackjax anesthetic fgivenx matplotlib numpy scipy
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from models import get_model, ModelConfig
from inference.blackjax_nested_sampling import BlackJAXNestedSampler

# Create model
model = get_model('emg', ModelConfig(max_peaks=2, fit_pulses=False))

# Generate synthetic data
t = jnp.linspace(0, 5, 100)
true_params = {
    'A': jnp.array([0.08, 0.06]),
    'tau': jnp.array([0.5, 0.3]),
    'u': jnp.array([1.5, 3.0]),
    'w': jnp.array([0.2, 0.15]),
    'sigma': jnp.array([0.003])
}

# Run nested sampling
sampler = BlackJAXNestedSampler(model, data, prior_config)
results = sampler.run(key, num_live_points=500, max_samples=50000)
```

## Testing

```bash
# Run tests
pytest tests/test_nested_sampling.py -v

# Run example scripts
python tests/2peak_fixed_test.py
python tests/2peak_fitpulses_test.py
```

## Model Types

- **EMG**: Exponential Modified Gaussian for FRB pulses
- **Exponential**: Simple exponential decay model
- **Periodic**: For periodic signals
- **Combined**: Combines multiple model types

## Key Features

### Continuous Npulse
The number of pulses is sampled as a continuous parameter for proper model selection, only discretized where necessary for computations.

### Proper Convergence
Nested sampling runs until the convergence criterion is met (log(Z_live) - log(Z) < -precision_criterion) rather than stopping at a fixed iteration count.

### GPU Support
Automatic GPU detection and memory management with configurable memory fraction.

## Citation

If you use this code, please cite the original FRBayes paper and acknowledge the JAX implementation.

## License

MIT License