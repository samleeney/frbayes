# FRBayes JAX

Fast Radio Burst Bayesian Analysis using JAX and BlackJAX nested sampling.

## Features

- **JAX-based Models**: Fully JIT-compilable EMG and Exponential pulse models
- **BlackJAX Nested Sampling**: Efficient Bayesian inference using BlackJAX's nested sampling implementation
- **Model Selection**: Can fit or fix the number of pulses
- **Visualization**: Integration with anesthetic and fgivenx for posterior analysis

## Installation

```bash
# Install dependencies
pip install jax jaxlib blackjax anesthetic fgivenx matplotlib numpy scipy pyyaml tqdm

# Install BlackJAX with nested sampling support
pip install git+https://github.com/handley-lab/blackjax@nested_sampling
```

## Usage

### Basic Example

```python
from frbayes_jax.models import get_model_function
from frbayes_jax.sampling import run_nested_sampling
from frbayes_jax.utils import get_default_prior_ranges

# Settings
model_name = "emg"  # or "exponential"
max_peaks = 2
fit_pulses = False  # Set to True to fit the number of pulses

# Get prior ranges
prior_ranges = get_default_prior_ranges(model_name)

# Run nested sampling
results = run_nested_sampling(
    model_name=model_name,
    data=data,  # Your pulse profile data
    t=time_axis,  # Time axis
    prior_ranges=prior_ranges,
    max_peaks=max_peaks,
    fit_pulses=fit_pulses,
    num_live_points=500,
    num_delete=25,
    num_inner_steps=10,
    verbose=True
)

print(f"Log evidence: {results['logZ']:.2f}")
```

### Running Tests

```bash
cd tests

# Test with fixed number of pulses
python test_2pulses_fixed.py

# Test with fitted number of pulses
python test_2pulses_fitted.py

# Run verification tests
python test_verification.py
```

## Models

- **EMG**: Exponentially Modified Gaussian model
- **Exponential**: Simple exponential decay model
- Both models support:
  - Multiple pulses
  - Baseline offset (optional)
  - Fixed or fitted number of pulses

## Structure

```
frbayes_jax/
├── frbayes_jax/
│   ├── __init__.py
│   ├── models.py       # JAX-based pulse models
│   ├── priors.py       # Prior transformations
│   ├── sampling.py     # BlackJAX nested sampling
│   ├── data.py         # Data preprocessing
│   ├── analysis.py     # Visualization and analysis
│   └── utils.py        # Utility functions
└── tests/
    ├── test_2pulses_fixed.py    # Test with fixed Npulse
    ├── test_2pulses_fitted.py   # Test with fitted Npulse
    └── test_verification.py     # Verification suite
```

## Citation

If you use this code, please cite:
- BlackJAX: Cabezas et al. (2024)
- Nested Sampling Implementation: Yallup and Handley (2025) [pending]