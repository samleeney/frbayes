# FRBayes

Fast Radio Burst Bayesian Analysis - JAX Implementation

## Overview

This repository contains a JAX-based implementation of FRBayes for analyzing Fast Radio Burst (FRB) pulse profiles using Bayesian nested sampling. The implementation leverages JAX for automatic differentiation and JIT compilation, and BlackJAX for efficient nested sampling.

## Main Implementation

The primary implementation is in `frbayes_jax/` which provides:

- **High-performance models**: JIT-compiled pulse models (EMG, Exponential)
- **Bayesian inference**: BlackJAX nested sampling for parameter estimation
- **Model selection**: Automatic determination of the number of pulses
- **Visualization**: Integration with anesthetic and fgivenx for posterior analysis

## Quick Start

```bash
# Install dependencies
pip install jax jaxlib blackjax anesthetic fgivenx matplotlib numpy scipy pyyaml tqdm
pip install git+https://github.com/handley-lab/blackjax@nested_sampling

# Run tests
cd frbayes_jax/tests
python test_verification.py
```

## Directory Structure

```
frbayes/
├── frbayes_jax/        # Main JAX implementation
│   ├── frbayes_jax/    # Package modules
│   ├── tests/          # Test suite
│   └── README.md       # Detailed documentation
└── archive/            # Archived files and older implementations
```

## Features

- **Multiple pulse models**: EMG (Exponentially Modified Gaussian) and Exponential
- **Flexible inference**: Fix or fit the number of pulses
- **JAX acceleration**: Fully JIT-compilable for fast execution
- **Robust sampling**: BlackJAX nested sampling with proper prior handling

## Tests

The implementation includes comprehensive tests:

1. **Fixed number of pulses**: Tests parameter recovery with known number of pulses
2. **Fitted number of pulses**: Tests model selection capability
3. **Verification suite**: Automated testing of both scenarios

All tests pass successfully, demonstrating correct implementation.

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:
- BlackJAX: Cabezas et al. (2024)
- Nested Sampling: Handley et al.