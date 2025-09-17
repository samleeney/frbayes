# FRBayes JAX Examples

This directory contains comprehensive examples demonstrating various use cases and configurations of the FRBayes JAX framework. Each example showcases different aspects of FRB pulse modeling and Bayesian inference.

For detailed system architecture, refer to the [Architecture Documentation](../../docs/ARCHITECTURE.md).

## Example Categories

### Basic Pulse Fitting Examples

#### 1. `test_2pulses_fitted.py`
**Model**: EMG (Exponentially Modified Gaussian)
**Configuration**: 2 pulses, fitted Npulse
**Purpose**: Demonstrates basic multi-pulse fitting with variable number of pulses

**Key Features**:
- Uses [`emg_model`](../../docs/ARCHITECTURE.md#2-modelspy---pulse-model-definitions) from models.py
- Implements sorted arrival time priors via [`FRBPriors`](../../docs/ARCHITECTURE.md#3-priorspy---prior-distributions)
- Shows basic nested sampling with [`run_nested_sampling`](../../docs/ARCHITECTURE.md#4-samplingpy---nested-sampling-engine)
- Generates corner plots and functional posteriors

**Use Case**: When the exact number of pulses is unknown and needs to be inferred from data.

#### 2. `test_2pulses_fixed.py`
**Model**: EMG
**Configuration**: 2 pulses, fixed Npulse
**Purpose**: Demonstrates fitting with known number of pulses

**Key Features**:
- Fixed model complexity (Npulse = 2)
- Faster convergence than fitted version
- Useful for hypothesis testing

**Use Case**: When the number of pulses is known a priori or for model comparison.

#### 3. `test_2pulses_exponential_fitted.py`
**Model**: Exponential decay
**Configuration**: 2 pulses, fitted Npulse
**Purpose**: Alternative pulse shape modeling

**Key Features**:
- Uses simpler [`exponential_model`](../../docs/ARCHITECTURE.md#2-modelspy---pulse-model-definitions)
- No width parameter (sharper rise time)
- Comparison with EMG results

**Use Case**: For FRBs with sharp rise times and exponential decay.

#### 4. `test_2pulses_exponential_fixed.py`
**Model**: Exponential decay
**Configuration**: 2 pulses, fixed Npulse
**Purpose**: Fixed exponential model fitting

**Key Features**:
- Combines exponential shape with fixed complexity
- Fastest configuration for simple models

### Complex Multi-Pulse Examples

#### 5. `test_6pulses_fitted.py`
**Model**: EMG
**Configuration**: Up to 6 pulses, fitted Npulse
**Purpose**: Moderate complexity pulse train modeling

**Key Features**:
- Tests scalability to more pulses
- Demonstrates prior handling for many parameters
- Shows convergence behavior with increased complexity

**Use Case**: FRBs with moderate complexity pulse structures.

#### 6. `test_7pulses_fitted.py`
**Model**: EMG
**Configuration**: Up to 7 pulses, fitted Npulse
**Purpose**: Higher complexity modeling

**Key Features**:
- Extended parameter space exploration
- Memory management considerations
- Visualization of complex posteriors

#### 7. `test_7pulses_exponential_fitted.py`
**Model**: Exponential
**Configuration**: Up to 7 pulses, fitted Npulse
**Purpose**: Complex exponential pulse trains

**Key Features**:
- Comparison with EMG for many pulses
- Computational efficiency testing

#### 8. `test_9pulses_fitted.py`
**Model**: EMG
**Configuration**: Up to 9 pulses, fitted Npulse
**Purpose**: Maximum complexity demonstration

**Key Features**:
- Tests framework limits
- Optimized for FRB 20191221A (known to have ~9 pulses)
- Extensive parameter space (36+ dimensions)

**Use Case**: Complex repeating FRBs with many sub-pulses.

### Advanced Models

#### 9. `test_periodic_exponential.py`
**Model**: Periodic exponential
**Configuration**: Regularly spaced pulses
**Purpose**: Models FRBs with periodic structure

**Key Features**:
- Uses [`periodic_exponential_model`](../../docs/ARCHITECTURE.md#2-modelspy---pulse-model-definitions)
- Only fits period and first pulse location
- Reduces parameter space for periodic signals
- Includes baseline offset option

**Use Case**: FRBs with clear periodic modulation like FRB 20191221A.

### 2D Spectral Analysis

#### 10. `test_2d_spectral.py`
**Model**: 2D EMG with spectral index
**Configuration**: Frequency-dependent modeling
**Purpose**: Full waterfall analysis with spectral evolution

**Key Features**:
- Uses [`emg_model_2d`](../../docs/ARCHITECTURE.md#2-modelspy---pulse-model-definitions)
- Fits spectral index α for frequency scaling
- Processes 2D waterfall data via [`preprocess_data_2d`](../../docs/ARCHITECTURE.md#1-datapy---data-preprocessing-and-simulation)
- Per-channel noise weighting

**Technical Details**:
- Models intensity as: `I(ν,t) = I₀(t) × (ν/ν_ref)^α`
- Handles dedispersed data
- Supports baseline models

**Use Case**: Studying frequency-dependent behavior and spectral evolution.

#### 11. `test_2d_rfi_mitigation.py`
**Model**: 2D with RFI mitigation
**Configuration**: Bayesian anomaly detection
**Purpose**: Robust fitting in presence of RFI

**Key Features**:
- Implements Bayesian anomaly detection
- Uses mixture model likelihood
- Robust to outliers and RFI contamination
- Adaptive threshold based on data

**Algorithm**:
```python
L = max(L_normal × (1-p), L_uniform × p)
```
where p is the anomaly probability.

**Use Case**: Real data with RFI contamination.

### Real Data Analysis

#### 12. `test_real_data.py`
**Model**: Configurable
**Configuration**: Real FRB data processing
**Purpose**: Complete pipeline for actual FRB analysis

**Key Features**:
- Full data preprocessing pipeline
- Multiple preprocessing modes (default, paper, raw)
- S/N calculation and optimization
- Complete visualization suite
- Model comparison capabilities

**Pipeline Steps**:
1. Load HDF5 waterfall data
2. Apply RFI mitigation
3. Downsample to target resolution
4. Extract pulse profile
5. Run nested sampling
6. Generate all visualizations
7. Save results

**Use Case**: Analyzing real FRB observations.

### Utility Examples

#### 13. `test_sorted_prior.py`
**Purpose**: Demonstrates sorted arrival time priors

**Key Features**:
- Tests [`forced_identifiability_transform`](../../docs/ARCHITECTURE.md#3-priorspy---prior-distributions)
- Validates prior sampling
- Ensures parameter ordering

**Technical Details**:
- Prevents label switching problem
- Maintains u₁ < u₂ < ... < uₙ constraint
- Uses special transformation for uniform sampling

#### 14. `test_sorted_sampling.py`
**Purpose**: Tests sampling with sorting constraints

**Key Features**:
- Validates nested sampling with constraints
- Tests prior boundary conditions
- Ensures proper convergence

#### 15. `plot_corner.py`
**Purpose**: Standalone corner plot generation

**Key Features**:
- Loads existing chains
- Customizable parameter selection
- Publication-quality figures
- Model-specific styling

## Running Examples

### Basic Usage
```bash
# Run a simple 2-pulse example
python frbayes_jax/examples/test_2pulses_fitted.py

# Run with custom parameters
python frbayes_jax/examples/test_6pulses_fitted.py \
    --num_live_points 2000 \
    --max_peaks 6 \
    --seed 42
```

### Common Parameters
Most examples accept these command-line arguments:
- `--num_live_points`: Number of live points for nested sampling (default: 1000)
- `--max_peaks`: Maximum number of pulses to fit
- `--fit_pulses`: Whether to fit Npulse (true/false)
- `--seed`: Random seed for reproducibility
- `--output_dir`: Directory for saving results

### Output Files
Each example generates:
- `chains/`: Nested sampling chains in anesthetic format
- `results/`: Visualization outputs
  - `corner_plot.png`: Parameter correlations
  - `functional_posterior.png`: Model predictions
  - `params_*.png`: 1D parameter distributions
  - `npulse_distribution.png`: Pulse count posterior (if fitted)

## Performance Considerations

### Memory Usage
- 2-4 pulse examples: ~1-2 GB RAM
- 6-7 pulse examples: ~4-6 GB RAM
- 9 pulse examples: ~8-10 GB RAM
- 2D examples: ~6-8 GB RAM

### Runtime Estimates
On a modern CPU (no GPU):
- 2 pulses: ~5-10 minutes
- 6 pulses: ~30-60 minutes
- 9 pulses: ~2-4 hours
- 2D models: ~1-2 hours

With GPU acceleration:
- 2-5x speedup for large models
- Requires JAX GPU support

### Optimization Tips
1. Start with fewer live points for testing
2. Use fixed Npulse for faster convergence
3. Enable GPU with `JAX_PLATFORM_NAME=gpu`
4. Adjust `log_tolerance` for speed vs accuracy

## Extending Examples

### Adding New Examples
1. Copy a similar template example
2. Modify model configuration
3. Adjust prior bounds as needed
4. Update visualization settings
5. Document purpose and use case

### Custom Models
To use custom models in examples:
1. Define model in `models.py`
2. Register in `get_model_function()`
3. Update prior bounds in example
4. Adjust parameter names for plots

## Troubleshooting

### Common Issues

**Out of Memory**:
- Reduce `num_live_points`
- Use chunked finalization
- Clear JAX cache between runs

**Slow Convergence**:
- Increase `num_inner_steps`
- Adjust prior bounds
- Check data preprocessing

**Plotting Errors**:
- Verify chain files exist
- Check for NaN values
- Ensure parameter names match

**JAX Errors**:
- Update JAX/jaxlib versions
- Check CUDA compatibility (GPU)
- Verify array types (jnp vs np)

## Scientific Applications

These examples support various scientific investigations:

1. **Pulse Component Analysis**: Decompose complex FRBs into individual pulses
2. **Model Selection**: Compare EMG vs exponential models via Bayesian evidence
3. **Periodicity Detection**: Identify and characterize periodic structures
4. **Spectral Analysis**: Study frequency-dependent properties
5. **RFI Mitigation**: Robust fitting in noisy environments
6. **Population Studies**: Systematic analysis of multiple FRBs

## Citations

If using these examples for research, please cite:
- FRBayes framework paper
- BlackJAX nested sampling
- Relevant FRB papers for specific models