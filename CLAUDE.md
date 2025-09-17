# FRBayes JAX

A high-performance Bayesian inference framework for modeling Fast Radio Bursts (FRBs) using JAX and nested sampling.

## 🎯 Project Overview

FRBayes JAX is designed to perform rigorous Bayesian analysis of FRB 20191221A, a repeating fast radio burst with sub-second periodicity. The framework enables:

- **Model Selection**: Quantitative comparison between different pulse models using Bayesian evidence
- **Pulse Decomposition**: Automated inference of the number and properties of sub-pulses
- **Advanced Models**: Support for exponentially modified Gaussian (EMG), exponential, and periodic models
- **2D Spectral Analysis**: Frequency-dependent modeling with spectral index fitting
- **High Performance**: JAX-based implementation with JIT compilation and GPU acceleration

For detailed project goals and scientific motivation, see [📋 Project Goals](docs/PROJECT_GOAL.md).

## 📚 Documentation Structure

### Core Documentation
- **[Project Goals](docs/PROJECT_GOAL.md)**: Scientific objectives and research roadmap
- **[Architecture](docs/ARCHITECTURE.md)**: Complete system design, module descriptions, and data flow
- **[Changelog](docs/changelogs/)**: Development history and recent updates
  - [Recent Changes](docs/changelogs/20250117_extensive_testing_implementation.md)
  - [Changelog Guidelines](docs/changelogs/README.md)

### Code Documentation
- **[Tests README](frbayes_jax/tests/README.md)**: Unit test descriptions and testing guide
- **[Examples README](frbayes_jax/examples/README.md)**: Comprehensive usage examples and tutorials

## 🏗️ Architecture Summary

The framework consists of six core modules working in a pipeline:

```
Data Loading → Preprocessing → Model Definition → Prior Setup → Nested Sampling → Analysis
```

### Key Components

1. **[`data.py`](docs/ARCHITECTURE.md#1-datapy---data-preprocessing-and-simulation)**: Data I/O and preprocessing
   - HDF5 waterfall loading
   - RFI mitigation strategies
   - Downsampling and S/N optimization

2. **[`models.py`](docs/ARCHITECTURE.md#2-modelspy---pulse-model-definitions)**: JAX-based pulse models
   - EMG and exponential pulses
   - Multi-pulse composition
   - 2D models with spectral evolution

3. **[`priors.py`](docs/ARCHITECTURE.md#3-priorspy---prior-distributions)**: Prior distributions
   - Sorted arrival times for identifiability
   - Configurable parameter bounds
   - Support for discrete parameters

4. **[`sampling.py`](docs/ARCHITECTURE.md#4-samplingpy---nested-sampling-engine)**: Bayesian inference engine
   - BlackJAX nested sampling integration
   - Memory-efficient algorithms
   - 2D and RFI-robust variants

5. **[`analysis.py`](docs/ARCHITECTURE.md#5-analysispy---results-visualization-and-analysis)**: Visualization and results
   - Corner plots with anesthetic
   - Functional posteriors with fgivenx
   - Statistical summaries

For complete architectural details, see [📐 Architecture Documentation](docs/ARCHITECTURE.md).

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/frbayes_fresh.git
cd frbayes_fresh

# Install dependencies
pip install jax jaxlib blackjax anesthetic fgivenx matplotlib numpy scipy pyyaml tqdm distrax h5py
pip install git+https://github.com/handley-lab/blackjax@nested_sampling

# Optional: Install GPU support
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Basic Usage

```python
from frbayes_jax import run_nested_sampling, analyze_results

# Run Bayesian inference
results = run_nested_sampling(
    model_name="emg",
    data=pulse_profile,
    t=time_axis,
    max_peaks=9,
    fit_pulses=True,
    num_live_points=1000
)

# Analyze and visualize
analyze_results(results, "emg", time_axis, pulse_profile)
```

### Example Scripts

Explore the [examples directory](frbayes_jax/examples/) for comprehensive demonstrations:

- **Basic fitting**: `test_2pulses_fitted.py` - Simple multi-pulse modeling
- **Complex models**: `test_9pulses_fitted.py` - FRB 20191221A analysis
- **Periodic models**: `test_periodic_exponential.py` - Periodic pulse trains
- **2D analysis**: `test_2d_spectral.py` - Spectral evolution modeling
- **Real data**: `test_real_data.py` - Complete analysis pipeline

See [📖 Examples Documentation](frbayes_jax/examples/README.md) for detailed descriptions.

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
pytest frbayes_jax/tests/

# Run specific test module
pytest frbayes_jax/tests/test_models.py -v

# Generate coverage report
pytest --cov=frbayes_jax --cov-report=html
```

See [🔬 Testing Documentation](frbayes_jax/tests/README.md) for test descriptions.

## 📊 Key Features

### Model Types
- **Exponentially Modified Gaussian (EMG)**: Realistic pulse shapes with rise and decay
- **Exponential**: Simple decay models with sharp rise
- **Periodic**: Regularly spaced pulses with fixed period
- **2D Spectral**: Frequency-dependent models with power-law scaling

### Analysis Capabilities
- **Model Selection**: Bayesian evidence for model comparison
- **Parameter Inference**: Full posterior distributions for all parameters
- **Pulse Counting**: Automatic determination of pulse numbers
- **RFI Mitigation**: Bayesian anomaly detection for robust fitting

### Performance
- **JAX Integration**: JIT compilation and automatic differentiation
- **GPU Acceleration**: Optional CUDA support for large-scale problems
- **Memory Optimization**: Chunked processing for large datasets
- **Parallel Sampling**: Efficient nested sampling with BlackJAX

## 📈 Current Development Status

The project is actively being developed with recent implementations including:

- ✅ Extensive test suite coverage
- ✅ Multiple pulse model variants
- ✅ 2D spectral analysis capabilities
- ✅ RFI mitigation strategies
- 🔄 Physical interpretation framework (in progress)
- 📝 Comprehensive documentation

See [📰 Recent Changes](docs/changelogs/20250117_extensive_testing_implementation.md) for the latest updates.

## Directory Structure

```
frbayes_fresh/
├── docs/                 # Documentation
│   ├── PROJECT_GOAL.md   # Scientific objectives
│   ├── ARCHITECTURE.md   # System design
│   └── changelogs/       # Development history
├── frbayes_jax/          # Main package
│   ├── frbayes_jax/      # Core modules
│   │   ├── data.py       # Data processing
│   │   ├── models.py     # Pulse models
│   │   ├── priors.py     # Prior distributions
│   │   ├── sampling.py   # Nested sampling
│   │   ├── analysis.py   # Visualization
│   │   └── utils.py      # Utilities
│   ├── tests/            # Unit tests
│   └── examples/         # Usage examples
└── README.md             # This file
```

## 🤝 Contributing

We welcome contributions! Please:

1. Check existing [issues](https://github.com/yourusername/frbayes_fresh/issues)
2. Follow the code style and architecture patterns
3. Add tests for new functionality
4. Update documentation as needed
5. Follow the [changelog guidelines](docs/changelogs/README.md)

## 📖 Citations

If you use FRBayes JAX in your research, please cite:

```bibtex
@software{frbayes_jax,
  title = {FRBayes JAX: Bayesian Inference for Fast Radio Bursts},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/frbayes_fresh}
}
```

Additionally, please cite:
- BlackJAX: Cabezas et al. (2024)
- Nested Sampling: Handley et al. (2015)
- FRB 20191221A: CHIME/FRB Collaboration (2022)

## 📜 License

[Add your license here]

## 🔗 Related Projects

- [BlackJAX](https://github.com/blackjax-devs/blackjax): Probabilistic programming in JAX
- [anesthetic](https://github.com/handley-lab/anesthetic): Nested sampling visualization
- [fgivenx](https://github.com/handley-lab/fgivenx): Functional posterior plotting

## 📞 Contact

[Add contact information]

---

*For detailed documentation on any component, follow the links throughout this README or explore the [docs/](docs/) directory.*