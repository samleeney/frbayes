# FRBayes JAX Test Suite

This directory contains unit tests for the core functionality of FRBayes JAX. These tests ensure the reliability and correctness of the pulse models, fitting procedures, and visualization components.

For a complete understanding of the system architecture, see the [Architecture Documentation](../../docs/ARCHITECTURE.md).

## Test Files

### 1. `test_models.py`
**Purpose**: Unit tests for all pulse model implementations defined in [`models.py`](../../docs/ARCHITECTURE.md#2-modelspy---pulse-model-definitions).

**Tests Coverage**:
- **Single Pulse Models**:
  - EMG (Exponentially Modified Gaussian) pulse shape validation
  - Exponential pulse shape validation
  - Parameter sensitivity testing

- **Multi-Pulse Models**:
  - Sum of multiple EMG pulses
  - Sum of multiple exponential pulses
  - Variable number of pulses (Npulse parameter)

- **Model Variants**:
  - Models with baseline offset
  - Periodic models with regular spacing
  - 2D models with spectral index

- **Parameter Management**:
  - `get_num_params()` correctness
  - `get_param_names()` formatting
  - Index retrieval functions (sigma, Npulse, spectral)

**Key Assertions**:
- Model outputs are JAX arrays with correct shapes
- JIT compilation works without errors
- Parameter counts match model specifications
- Spectral scaling behaves correctly for 2D models

### 2. `test_fittedfixed.py`
**Purpose**: Tests the sampling framework's ability to handle both fitted and fixed parameter scenarios, interfacing with [`sampling.py`](../../docs/ARCHITECTURE.md#4-samplingpy---nested-sampling-engine).

**Tests Coverage**:
- **Fixed Npulse Tests**:
  - Sampling with predetermined number of pulses
  - Prior bounds enforcement
  - Convergence with fixed model complexity

- **Fitted Npulse Tests**:
  - Variable number of pulses inference
  - Model selection via Bayesian evidence
  - Prior handling for discrete parameters

- **Comparison Tests**:
  - Evidence comparison between fixed/fitted modes
  - Parameter recovery accuracy
  - Computational efficiency differences

**Key Validations**:
- Nested sampling convergence criteria
- Proper handling of mixed continuous/discrete parameters
- Memory efficiency during sampling
- BlackJAX integration stability

### 3. `test_plotting.py`
**Purpose**: Validates the visualization pipeline implemented in [`analysis.py`](../../docs/ARCHITECTURE.md#5-analysispy---results-visualization-and-analysis).

**Tests Coverage**:
- **Corner Plots**:
  - Parameter correlation visualization
  - Anesthetic integration
  - Model-specific color schemes
  - Parameter subset selection

- **Functional Posteriors**:
  - fgivenx integration
  - Confidence band calculations
  - Model realization sampling
  - Data overlay functionality

- **Parameter Distributions**:
  - 1D marginal plots
  - Parameter grouping (amplitudes, tau, arrival times)
  - Npulse distribution histograms
  - Statistical summary generation

**Key Checks**:
- Plot generation without errors
- File I/O for saved figures
- NaN/Inf handling in chains
- Proper weight calculations from nested sampling

## Running the Tests

### Prerequisites
Ensure all dependencies are installed:
```bash
pip install pytest jax jaxlib blackjax distrax anesthetic fgivenx
```

### Running Individual Test Files
```bash
# Test models
pytest frbayes_jax/tests/test_models.py -v

# Test fitting procedures
pytest frbayes_jax/tests/test_fittedfixed.py -v

# Test plotting
pytest frbayes_jax/tests/test_plotting.py -v
```

### Running All Tests
```bash
# Run all tests in the directory
pytest frbayes_jax/tests/ -v

# With coverage report
pytest frbayes_jax/tests/ --cov=frbayes_jax --cov-report=html
```

### Test Markers and Categories
Tests can be marked for selective execution:
- `@pytest.mark.slow` - Long-running sampling tests
- `@pytest.mark.gpu` - Tests requiring GPU acceleration
- `@pytest.mark.plotting` - Visualization tests

Run specific categories:
```bash
# Skip slow tests
pytest -m "not slow"

# Only run GPU tests
pytest -m gpu
```

## Test Data

The tests use synthetic data generated via [`data.py`](../../docs/ARCHITECTURE.md#1-datapy---data-preprocessing-and-simulation)'s `simulate_frb_data()` function with known parameters to validate:
- Parameter recovery accuracy
- Model selection capabilities
- Noise handling
- Prior constraint enforcement

## Integration with CI/CD

These tests are designed to be run in continuous integration environments:
- Compatible with GitHub Actions
- Memory-efficient for cloud runners
- Deterministic with fixed random seeds
- Platform-independent (Linux/macOS/Windows)

## Common Test Patterns

### Model Testing Pattern
```python
def test_model_output_shape():
    """Test that model returns correct shape"""
    model_func = get_model_function("emg")
    t = jnp.linspace(0, 4, 100)
    theta = sample_prior()
    output = model_func(t, theta, max_peaks=2, fit_pulses=False)
    assert output.shape == t.shape
```

### Sampling Testing Pattern
```python
def test_nested_sampling_convergence():
    """Test that sampling converges properly"""
    data = generate_synthetic_data()
    result = run_nested_sampling(
        model_name="emg",
        data=data,
        t=time_axis,
        num_live_points=100,
        log_tolerance=-3.0
    )
    assert result.logZ is not None
    assert len(result.particles) > 0
```

### Visualization Testing Pattern
```python
def test_corner_plot_generation():
    """Test corner plot creation"""
    chains = load_test_chains()
    fig = plot_corner(
        chains,
        param_names=get_param_names("emg", 2, False)
    )
    assert fig is not None
    assert len(fig.axes) > 0
```

## Debugging Tips

1. **Memory Issues**: Reduce `num_live_points` in sampling tests
2. **JAX Errors**: Check for proper array types (use `jnp` not `np`)
3. **Plotting Failures**: Verify matplotlib backend settings
4. **Slow Tests**: Use `@pytest.mark.slow` and skip during development

## Contributing New Tests

When adding new functionality:
1. Write tests before implementation (TDD approach)
2. Cover edge cases and error conditions
3. Use descriptive test names
4. Add appropriate markers for test categories
5. Document expected behavior in docstrings
6. Ensure tests are deterministic (fixed seeds)