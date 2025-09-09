# FRBayes CPU Module - Comprehensive Review

## Overview

FRBayes is a Bayesian inference framework for analyzing Fast Radio Burst (FRB) data, specifically designed for repeater FRBs that exhibit multiple pulse structures. The system uses nested sampling (via PolyChord) to fit various pulse models to observed FRB data, with the capability to determine the optimal number of pulses and their parameters.

**Key Features:**
- Multiple pulse models (EMG, Exponential, Periodic variants)
- Flexible number of pulses (can be fitted as a parameter)
- Nested sampling for Bayesian inference
- Comprehensive visualization and analysis tools
- Configurable preprocessing pipelines

## Architecture and File Structure

### Core Entry Point

#### `main.py`
The main execution script that orchestrates the entire FRB analysis pipeline:
- **Environment Variable Handling**: Reads optional model configuration from environment
- **Settings Management**: Loads global settings from YAML configuration
- **Pipeline Execution**:
  1. Data preprocessing via `data.preprocess_data()`
  2. Analysis initialization with `FRBAnalysis()`
  3. Model setup with `FRBModel()`
  4. PolyChord nested sampling execution
  5. Post-processing: chain analysis, functional posteriors, overlay predictions

**Optional environment variables:**
- `MODEL_FRB`: Overrides model type
- `PREPROCESSING_MODE`: Sets preprocessing approach

### Configuration System

#### `settings.yaml`
Comprehensive configuration file controlling all aspects of the analysis:

**Data Configuration:**
- `data_file`: Path to HDF5 input data
- `original_freq_res`: 24414.0625 Hz (raw frequency resolution)
- `original_time_res`: 0.98304e-3 s (raw time resolution)
- `desired_freq_res`: 3.125e6 Hz (target frequency resolution)
- `desired_time_res`: 7.86432e-3 s (target time resolution)
- `freq_min/max`: 400-800 MHz frequency range

**Preprocessing Modes:**
1. **"default"**: Standard preprocessing with NaN→0 replacement and downsampling
2. **"paper"**: Reproduces published methodology with off-burst baseline subtraction
3. **"raw"**: Minimal processing, no downsampling

**Model Configuration:**
- `model`: Selected model type (emg, exponential, periodic variants)
- `max_peaks`: Maximum number of pulses to fit (1-10 typically)
- `fit_pulses`: Boolean to fit number of pulses as parameter

**Prior Ranges:** Extensive configuration for each model's parameter priors
- Common priors: amplitude, tau, u (arrival time), sigma
- Model-specific: width (EMG), baseline offset, period parameters

#### `frbayes/settings.py`
Settings management class that:
- Loads YAML configuration
- Provides fallback defaults for missing values
- Manages prior ranges with model-specific overrides
- Global singleton instance `global_settings`

### Model Definitions

#### `frbayes/models.py`
Contains all pulse model implementations with base class hierarchy:

**Base Model Class (`BaseModel`):**
- Abstract interface for all models
- Defines `model_function()`, `loglikelihood()`, `prior()` methods
- Manages parameter names and dimensions

**1. EMGModel (Exponentially Modified Gaussian):**
- **Parameters per pulse**: 4 (A, τ, u, w)
- **Global parameters**: σ (noise), Npulse (if fitted)
- **Mathematical formulation:**
```
f(t) = (A/(2τ)) * exp((u-t)/τ + w²/(2τ²)) * erfc(((u-t)τ + w²)/(w*τ*√2))
```
- Most complex model with both Gaussian and exponential components
- Width parameter w controls pulse shape

**2. EMGModelWithBaseline:**
- EMG model + constant baseline offset parameter
- Useful for data with DC offset or systematic bias
- Additional parameter: B_offset

**3. ExponentialModel:**
- **Parameters per pulse**: 3 (A, τ, u)
- **Mathematical formulation:**
```
f(t) = A * exp(-(t-u)/τ) for t > u, else 0
```
- Simpler model for sharp rise, exponential decay pulses
- No width parameter

**4. ExponentialModelWithBaseline:**
- Exponential model + constant baseline offset

**5. PeriodicExponentialModel:**
- **Shared period T and initial phase u₀**
- **Mathematical formulation:**
```
u_n = u₀ + n*T  (arrival times)
f_n(t) = A_n * exp(-(t-u_n)/τ_n) for t > u_n
```
- Assumes regular periodic structure

**6. PeriodicEMGModel:**
- EMG variant with periodic structure
- Combines EMG pulse shape with periodic timing

**7. Complex Combined Models:**
- `PeriodicExponentialPlusExponentialModel`: Periodic + non-periodic components
- `DoublePeriodicExponentialModel`: Two independent periodic structures

**Prior Transformations:**
- Uses PyPolyChord priors (Uniform, LogUniform, SortedUniform)
- Arrival times use SortedUniformPrior for active pulses (ensures ordering)
- Inactive pulses (when Npulse < max_peaks) use standard uniform priors

### Sampling Infrastructure

#### `frbayes/sample.py`
Manages the Bayesian sampling process:
- **FRBModel Class**:
  - Interfaces between data, model, and PolyChord
  - Wraps model likelihood and prior functions
  - Configures PolyChord settings:
    - `num_repeats`: nDims * 10 (sampling efficiency)
    - `nlive`: nDims * 50 (number of live points)
    - `do_clustering`: True (enables mode separation)
    - `read_resume`: True (allows resuming interrupted runs)

### Data Processing

#### `frbayes/data.py`
Handles data loading and preprocessing:

**Preprocessing Pipelines:**

1. **"raw" mode**:
   - No NaN replacement
   - No downsampling
   - Direct mean for pulse profile

2. **"paper" mode**:
   - RFI mitigation: NaN → off-burst median
   - Downsampling to desired resolution
   - Baseline subtraction using first 10% of time bins
   - SNR calculation with noise from off-pulse region

3. **"default" mode**:
   - NaN → 0 replacement
   - Downsampling to desired resolution
   - Simple mean for pulse profile

**Output:**
- Downsampled waterfall data
- Pulse profile SNR
- Time axis array
- Saves to CSV for external analysis

#### `frbayes/utils.py`
Utility functions:
- `downsample()`: 2D array downsampling via reshaping and averaging
- `calculate_snr()`: SNR computation (currently unused in main pipeline)
- `load_settings()`: Legacy settings loader

### Analysis and Visualization

#### `frbayes/analysis.py`
Comprehensive post-processing and visualization:

**FRBAnalysis Class:**

1. **Input Visualization** (`plot_inputs()`):
   - Waterfall plot with model-specific colormap
   - Pulse profile SNR plot

2. **Functional Posteriors** (`functional_posteriors()`):
   - Uses fgivenx for posterior predictive distributions
   - Generates contour plots showing uncertainty bands
   - Handles combined models with separate component visualization

3. **Chain Processing** (`process_chains()`):
   - Corner plots for parameter correlations
   - Separate plots for amplitude, tau, arrival time, width parameters
   - Period distribution for periodic models

4. **Overlay Predictions** (`overlay_predictions_on_input()`):
   - Overlays model predictions on observed data
   - Shows individual components for combined models

**Color Schemes:**
- Each model has unique color and colormap
- EMG: Blue, Exponential: Green, Periodic: Red/Orange/Purple

### Simulation Capabilities

#### `frbayes/simulator.py`
Data simulation for testing:
- Generate synthetic FRB data from any model
- Add Gaussian noise
- Useful for algorithm validation and testing

## Mathematical Formulations

### 1. Exponentially Modified Gaussian (EMG)
The EMG combines a Gaussian with an exponential decay:
```
f(t) = (A/(2τ)) × exp((u-t)/τ + w²/(2τ²)) × erfc(((u-t)τ + w²)/(w×τ×√2))
```
Where:
- A: Amplitude
- τ: Exponential decay time constant
- u: Peak arrival time
- w: Gaussian width parameter
- erfc: Complementary error function

### 2. Exponential Model
Simple exponential decay after sharp rise:
```
f(t) = { A × exp(-(t-u)/τ)  if t > u
       { 0                   if t ≤ u
```

### 3. Periodic Models
For periodic structure with period T and initial phase u₀:
```
u_n = u₀ + n×T
```
Each pulse follows either EMG or Exponential profile centered at u_n.

### 4. Likelihood Function
Gaussian likelihood assuming independent noise:
```
log L = -0.5 × Σ((data - model)²/σ²) - N×log(σ×√(2π))
```
Where σ is the fitted noise standard deviation.

## Critical Implementation Details

### Npulse Handling
- **Continuous Sampling**: Npulse is sampled as a continuous parameter [1, max_peaks+1]
- **Discrete Application**: Rounded to integer when evaluating the model
- This approach avoids trans-dimensional sampling complications
- Inactive pulses (i > Npulse) don't contribute to likelihood

### Prior Ordering
- Active pulse arrival times use SortedUniformPrior
- Ensures u₁ < u₂ < ... < u_Npulse
- Prevents label switching and improves sampling efficiency

### Memory Efficiency
- Time array operations not vectorized due to size
- Explicit loops over pulses for model evaluation
- Careful management of large waterfall arrays

## Running the Code

### Basic Execution
```bash
python frbayes_cpu/main.py
```

### Environment Configuration
```bash
export MODEL_FRB="emg"
export PREPROCESSING_MODE="paper"
python frbayes_cpu/main.py
```

### Output Structure
```
chains/              # PolyChord output
  ├── fit_pulses=True_emg_npeaks=5.txt
  ├── fit_pulses=True_emg_npeaks=5.stats
  └── fit_pulses=True_emg_npeaks=5_dead-birth.txt
chains/results/      # Analysis outputs
  ├── inputs.png
  ├── *_posterior.png
  ├── *_f_posterior_combined.png
  └── *_overlay_predictions_on_snr.png
results/             # Data outputs
  ├── pulse_profile_snr.csv
  └── time_axis.csv
```

## Key Strengths and Design Choices

1. **Flexible Model Architecture**: Easy to add new models via inheritance
2. **Comprehensive Configuration**: Single YAML file controls entire pipeline
3. **Robust Sampling**: PolyChord handles multimodal posteriors well
4. **Production Ready**: Resume capability, extensive logging
5. **Visualization Suite**: Automatic generation of publication-quality plots
6. **Modular Design**: Clear separation of concerns across modules

## Potential Improvements

1. **Performance**: JAX/NumPyro port for GPU acceleration
2. **Model Selection**: Automatic Bayes factor computation
3. **Real-time Processing**: Streaming data capability
4. **Extended Models**: Time-varying width, frequency-dependent structure
5. **Documentation**: API documentation, user guide, tutorials