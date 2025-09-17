# FRBayes JAX Architecture

## Overview
FRBayes JAX is a Bayesian inference framework for modeling Fast Radio Bursts (FRBs) using JAX for high-performance computation. The codebase is organized into modular components that handle data preprocessing, model definition, prior specification, nested sampling, and result analysis.

## Core Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input & Data                        │
│              (FRB waterfall data in HDF5 format)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                       data.py                                │
│     • Data loading from HDF5                                 │
│     • Downsampling & preprocessing                           │
│     • RFI mitigation                                         │
│     • 1D/2D data preparation                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      models.py                               │
│     • JAX-based pulse models (EMG, Exponential)             │
│     • Multi-pulse composition                                │
│     • 2D models with spectral index                         │
│     • Periodic models                                        │
│     • Parameter management                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      priors.py                               │
│     • Prior distributions using distrax                      │
│     • Forced identifiability transforms                      │
│     • Sorted arrival time priors                             │
│     • RFI mitigation priors                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     sampling.py                              │
│     • BlackJAX nested sampling integration                   │
│     • Log-likelihood functions                               │
│     • Memory-efficient finalisation                          │
│     • 1D and 2D sampling pipelines                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     analysis.py                              │
│     • Anesthetic integration for visualization               │
│     • Corner plots                                           │
│     • Functional posterior plots (fgivenx)                   │
│     • Parameter distributions                                │
│     • Complete analysis pipeline                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      utils.py                                │
│     • Helper functions                                       │
│     • NaN handling for BlackJAX outputs                     │
└─────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. `data.py` - Data Preprocessing and Simulation
**Purpose**: Handles all data loading, preprocessing, and simulation tasks.

**Key Functions**:
- `downsample()`: Reduces data resolution by averaging adjacent bins
- `preprocess_data()`: Main 1D data preprocessing pipeline
  - Loads HDF5 waterfall data
  - Applies RFI mitigation (NaN replacement)
  - Downsamples to desired resolution
  - Computes S/N profiles
- `preprocess_data_2d()`: 2D data preprocessing for spectral analysis
  - Similar to 1D but preserves frequency dimension
  - Estimates per-channel noise
- `simulate_frb_data()`: Generates synthetic FRB data for testing

**Data Flow**:
- Input: Raw HDF5 waterfall files
- Output: Preprocessed numpy arrays ready for modeling

### 2. `models.py` - Pulse Model Definitions
**Purpose**: Defines all mathematical models for FRB pulses using JAX for automatic differentiation and JIT compilation.

**Model Types**:
1. **Basic Models**:
   - `emg_pulse()`: Single Exponentially Modified Gaussian pulse
   - `exponential_pulse()`: Single exponential decay pulse

2. **Multi-pulse Models**:
   - `emg_model()`: Sum of multiple EMG pulses
   - `exponential_model()`: Sum of multiple exponential pulses
   - Variants with baseline offsets

3. **Periodic Models**:
   - `periodic_exponential_model()`: Regularly spaced pulses with fixed period

4. **2D Models** (with spectral index):
   - `emg_model_2d()`: EMG with frequency-dependent scaling
   - `exponential_model_2d()`: Exponential with frequency scaling

**Key Features**:
- All models are `@jit` compiled for performance
- Support for variable number of pulses (Npulse parameter)
- Parameter management functions (`get_num_params()`, `get_param_names()`)
- Spectral index support for 2D fitting

**Parameter Structure**:
- EMG: `[A₁...Aₙ, τ₁...τₙ, u₁...uₙ, w₁...wₙ, (B_offset), (α), σ, (Npulse)]`
- Exponential: `[A₁...Aₙ, τ₁...τₙ, u₁...uₙ, (B_offset), (α), σ, (Npulse)]`

### 3. `priors.py` - Prior Distributions
**Purpose**: Manages prior probability distributions for all model parameters.

**Key Components**:
- `FRBPriors` class: Central prior management system
  - Configurable bounds for all parameter types
  - Support for different model types
  - Automatic dimension calculation

- `forced_identifiability_transform()`: Ensures sorted arrival times
  - Prevents label switching in multi-pulse models
  - Maintains parameter identifiability

**Prior Types**:
- Uniform priors: Amplitudes, decay times, widths
- Log-uniform priors: Noise parameter (σ)
- Sorted uniform: Arrival times (enforces u₁ < u₂ < ... < uₙ)
- Special priors: Spectral index, baseline, RFI anomaly probability

### 4. `sampling.py` - Nested Sampling Engine
**Purpose**: Implements the Bayesian inference using BlackJAX's nested sampling.

**Main Functions**:
- `run_nested_sampling()`: 1D model fitting pipeline
  - Sets up log-likelihood and log-prior functions
  - Manages BlackJAX nested sampling algorithm
  - Handles convergence criteria

- `run_nested_sampling_2d()`: 2D model fitting with spectral index
  - Extended likelihood for 2D waterfall data
  - Optional RFI mitigation via Bayesian anomaly detection
  - Per-channel noise weighting

- `finalise_chunked()`: Memory-efficient post-processing
  - Processes dead points in batches
  - Prevents out-of-memory errors for large runs
  - Custom implementation to handle BlackJAX's memory usage

**Key Features**:
- JIT-compiled likelihood and prior functions
- Progress tracking with tqdm
- Memory management and cache clearing
- NaN handling for BlackJAX outputs
- Support for sorted priors (arrival times)

### 5. `analysis.py` - Results Visualization and Analysis
**Purpose**: Provides comprehensive visualization and analysis tools for sampling results.

**Visualization Functions**:
- `plot_corner()`: Triangle plots using anesthetic
  - Shows parameter correlations
  - Model-specific coloring

- `plot_functional_posterior()`: Model predictions using fgivenx
  - Confidence bands for fitted models
  - Individual model realizations

- `plot_parameter_distributions()`: 1D marginal distributions
  - Grouped by parameter type
  - Special handling for Npulse distribution

- `analyze_results()`: Complete analysis pipeline
  - Orchestrates all visualization steps
  - Generates summary statistics
  - Saves results in anesthetic format

**Integration Points**:
- Reads BlackJAX sampling outputs
- Converts to anesthetic format for plotting
- Uses model functions from `models.py` for predictions

### 6. `utils.py` - Utility Functions
**Purpose**: Helper functions for common operations.

**Current Functions**:
- `fix_nan_logL_birth()`: Fixes NaN values in BlackJAX outputs
  - Required for anesthetic compatibility
  - Sets NaN birth likelihoods to maximum value

## Data Flow Pipeline

### 1. **Input Stage**
```
HDF5 File → data.py → Preprocessed Arrays
```
- Raw waterfall data loaded
- RFI mitigation applied
- Downsampling performed
- S/N profile extracted

### 2. **Model Setup Stage**
```
models.py + priors.py → Model Function + Prior Bounds
```
- Model function selected based on user choice
- Prior bounds configured
- Parameter dimensions calculated

### 3. **Inference Stage**
```
Data + Model + Priors → sampling.py → Posterior Samples
```
- Nested sampling initialized with live points
- Likelihood evaluated using JAX models
- Dead points accumulated until convergence
- Results finalized and cleaned

### 4. **Analysis Stage**
```
Posterior Samples → analysis.py → Plots & Statistics
```
- Samples converted to anesthetic format
- Corner plots generated
- Functional posteriors computed
- Summary statistics calculated

## Key Design Decisions

### 1. **JAX for Performance**
- All models JIT-compiled for speed
- Automatic differentiation available
- GPU acceleration possible
- Vectorized operations throughout

### 2. **Modular Architecture**
- Clear separation of concerns
- Each module has distinct responsibility
- Easy to extend with new models
- Minimal coupling between modules

### 3. **Memory Management**
- Custom chunked finalization for large runs
- Explicit cache clearing
- Batch processing of dead points
- Memory monitoring during sampling

### 4. **Sorted Priors for Identifiability**
- Forced ordering of arrival times
- Prevents label switching problem
- Maintains parameter interpretability
- Custom transform for uniform sampling

### 5. **Flexible Model System**
- Easy addition of new pulse shapes
- Support for periodic patterns
- 2D models with spectral information
- Variable number of pulses

## Extension Points

### Adding New Models
1. Define pulse function in `models.py`
2. Add to model registry in `get_model_function()`
3. Update parameter counting in `get_num_params()`
4. Add parameter names in `get_param_names()`
5. Update `FRBPriors` class for new parameters

### Adding New Preprocessing
1. Add preprocessing function to `data.py`
2. Define new preprocessing mode
3. Update data loading pipeline

### Adding New Visualizations
1. Add plotting function to `analysis.py`
2. Update color schemes if needed
3. Integrate into `analyze_results()` pipeline

## Dependencies and External Libraries

- **JAX**: Core computational framework
- **BlackJAX**: Nested sampling implementation
- **distrax**: Prior distributions
- **anesthetic**: Posterior visualization
- **fgivenx**: Functional posterior plotting
- **numpy/matplotlib**: Basic numerics and plotting
- **h5py**: HDF5 file I/O
- **tqdm**: Progress bars

## Performance Considerations

1. **JIT Compilation**: First call to models slower due to compilation
2. **Memory Usage**: Scales with `num_live_points × num_iterations`
3. **GPU Acceleration**: Available for large-scale problems
4. **Vectorization**: Models use vectorized operations where possible
5. **Chunked Processing**: Prevents memory overflow in post-processing