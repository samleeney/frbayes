# fgivenx

Python package for plotting posteriors of functions, providing visual representations of functional uncertainty in Bayesian analysis.

## Overview

fgivenx (pronounced "f given x") generates contour plots of functional posterior distributions. It's particularly useful in scientific Bayesian analysis for visualizing how a function's output varies probabilistically across its domain, given uncertainty in its parameters.

## Installation

```bash
pip install fgivenx
```

For development version:
```bash
git clone https://github.com/handley-lab/fgivenx
cd fgivenx
pip install -e .
```

### Optional Dependencies

For enhanced functionality:
```bash
pip install fgivenx[all]  # Installs all extras
```

Or individually:
- `joblib`: For parallel computation
- `tqdm`: For progress bars
- `getdist`: For compatibility with GetDist file formats

## Key Features

- Functional posterior contour plotting
- Kullback-Leibler divergence visualization
- Parallel computation support
- Caching for improved performance
- Integration with standard sampling packages

## Core Functions

- `plot_contours()`: Generate probability contour plots
- `plot_lines()`: Plot individual function realizations
- `plot_dkl()`: Compute and plot Kullback-Leibler divergence between distributions
- `samples_from_getdist()`: Load samples from GetDist format

## Complete Example: Linear Model with Uncertainty

```python
import numpy
import matplotlib.pyplot as plt
from fgivenx import plot_contours, plot_lines, plot_dkl

# Model definition
# ================
# Define a simple straight line function, parameters theta=(m,c)
def f(x, theta):
    m, c = theta
    return m * x + c

# Set random seed for reproducibility
numpy.random.seed(1)

# Generate posterior samples
# ==========================
nsamples = 1000

# Posterior: concentrated around m=-5, c=2
ms_posterior = numpy.random.normal(loc=-5, scale=1, size=nsamples)
cs_posterior = numpy.random.normal(loc=2, scale=1, size=nsamples)
samples = numpy.array([(m, c) for m, c in zip(ms_posterior, cs_posterior)]).copy()

# Prior: broader distribution around m=0, c=0
ms_prior = numpy.random.normal(loc=0, scale=5, size=nsamples)
cs_prior = numpy.random.normal(loc=0, scale=5, size=nsamples)
prior_samples = numpy.array([(m, c) for m, c in zip(ms_prior, cs_prior)]).copy()

# Define x-domain
# ===============
xmin, xmax = -2, 2
nx = 100
x = numpy.linspace(xmin, xmax, nx)

# Set cache directories for performance
cache = 'cache/test'
prior_cache = cache + '_prior'

# Create visualization
# ====================
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 1. Parameter Space Plot
# -----------------------
ax_samples = axes[0, 0]
ax_samples.set_ylabel(r'$c$')
ax_samples.set_xlabel(r'$m$')
ax_samples.set_title('Parameter Space')
ax_samples.plot(prior_samples.T[0], prior_samples.T[1], 'b.', alpha=0.3, label='Prior')
ax_samples.plot(samples.T[0], samples.T[1], 'r.', alpha=0.3, label='Posterior')
ax_samples.legend()

# 2. Function Realizations
# ------------------------
ax_lines = axes[0, 1]
ax_lines.set_ylabel(r'$y = m x + c$')
ax_lines.set_xlabel(r'$x$')
ax_lines.set_title('Function Realizations')
plot_lines(f, x, prior_samples, ax_lines, color='b', alpha=0.1, cache=prior_cache)
plot_lines(f, x, samples, ax_lines, color='r', alpha=0.1, cache=cache)

# 3. Functional Posterior Contours
# --------------------------------
ax_fgivenx = axes[1, 1]
ax_fgivenx.set_ylabel(r'$P(y|x)$')
ax_fgivenx.set_xlabel(r'$x$')
ax_fgivenx.set_title('Functional Posterior')

# Plot prior contours in blue
cbar = plot_contours(
    f, x, prior_samples, ax_fgivenx,
    colors=plt.cm.Blues_r, 
    lines=False,
    cache=prior_cache
)

# Plot posterior contours in default colors
cbar = plot_contours(f, x, samples, ax_fgivenx, cache=cache)

# 4. Kullback-Leibler Divergence
# ------------------------------
ax_dkl = axes[1, 0]
ax_dkl.set_ylabel(r'$D_\mathrm{KL}$')
ax_dkl.set_xlabel(r'$x$')
ax_dkl.set_title('KL Divergence: Posterior vs Prior')
ax_dkl.set_ylim(bottom=0, top=2.0)
plot_dkl(
    f, x, samples, prior_samples, ax_dkl,
    cache=cache, 
    prior_cache=prior_cache
)

# Link x-axes for synchronized zooming
ax_lines.get_shared_x_axes().join(ax_lines, ax_fgivenx, ax_dkl)

# Finalize plot
fig.tight_layout()
fig.savefig('functional_posterior_analysis.png', dpi=150)
plt.show()
```

## Advanced Usage

### Custom Number of Samples

Control the number of samples used for contour calculation:

```python
plot_contours(f, x, samples, ax, nsamples=500)  # Use subset of samples
```

### Parallel Computation

Enable parallel processing with joblib:

```python
plot_contours(f, x, samples, ax, parallel=True, nprocs=4)
```

### Custom Contour Levels

Specify confidence levels for contours:

```python
plot_contours(f, x, samples, ax, contours=[0.68, 0.95, 0.99])
```

### Working with GetDist

Load samples from GetDist chains:

```python
from fgivenx import samples_from_getdist
samples = samples_from_getdist("chains/chain", params=['param1', 'param2'])
```

## Use Cases

1. **Cosmology**: Visualizing uncertainty in theoretical predictions
2. **Regression Analysis**: Showing prediction intervals
3. **Model Comparison**: Comparing functional forms with uncertainty
4. **Sensitivity Analysis**: Understanding parameter influence on predictions

## Performance Tips

1. **Use Caching**: Specify cache directories to avoid recomputation
2. **Parallel Processing**: Enable parallel computation for large datasets
3. **Sample Thinning**: Use representative subsets for faster plotting
4. **Pre-computation**: Save computed caches for frequently used analyses

## Dependencies

### Required
- Python 2.7+ or 3.4+
- numpy
- scipy
- matplotlib

### Optional
- joblib (parallelization)
- tqdm (progress bars)
- getdist (file compatibility)

## Citation

If you use fgivenx in your research, please cite the relevant papers listed in the [GitHub repository](https://github.com/handley-lab/fgivenx).

## Links

- [GitHub Repository](https://github.com/handley-lab/fgivenx)
- [Documentation](https://fgivenx.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/fgivenx/)