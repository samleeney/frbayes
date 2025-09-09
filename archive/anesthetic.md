# Anesthetic

A Python library for processing and visualizing nested sampling chains by leveraging standard scientific Python libraries.

## Overview

Anesthetic brings together tools for post-processing nested sampling data, providing efficient computation of nested sampling statistics and intuitive plotting routines for marginal posteriors.

## Installation

```bash
git clone https://github.com/handley-lab/anesthetic.git
cd anesthetic
pip install .
```

Or via pip:
```bash
pip install anesthetic
```

## Key Features

- Load and process nested sampling chains
- Plot 1D and 2D marginal posteriors
- Compute nested sampling statistics (Bayesian evidence, model complexity)
- Support for prior/posterior comparison
- Integration with standard scientific Python stack

## Documentation

- [Quickstart Guide](https://anesthetic.readthedocs.io/en/latest/quickstart.html)
- [Plotting Tutorial](https://anesthetic.readthedocs.io/en/latest/plotting.html)
- [Full Documentation](https://anesthetic.readthedocs.io/)

## Basic Usage

### Reading Chains

```python
from anesthetic import read_chains

# Load nested sampling chains
# The file being loaded is [root]_dead-birth.txt
# In anesthetic, you only need: samples = read_chains([root])
samples = read_chains("chains/chains")  # Loads chains/chains_dead-birth.txt

# Get prior samples
prior = samples.prior()
```

**Important**: The only way to load samples in anesthetic is to use the `read_chains` function. Do not use `NestedSamples` or `MCMCSamples` as these are deprecated.

### Example 1: 1D Marginal Posteriors

```python
from anesthetic import read_chains, make_1d_axes

samples = read_chains("../../tests/example_data/pc")
params = ['x0', 'x1', 'x2', 'x3', 'x4']

# Create 1D axes grid
fig, axes = make_1d_axes(params, figsize=(8, 2))

# Plot posteriors
samples.plot_1d(axes)
```

### Example 2: 2D Marginal Posteriors

```python
from anesthetic import read_chains, make_2d_axes

# Define parameter names - these must match your chain columns
params = ['x0', 'x1', 'x2', 'x3', 'x4']

# Load chains with parameter names
samples = read_chains("../../tests/example_data/pc_250", columns=params)
prior = samples.prior()

# Create triangle plot - pass the parameter list
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w')

# Plot prior and posterior
prior.plot_2d(axes, alpha=0.9, label="prior")
samples.plot_2d(axes, alpha=0.9, label="posterior")

# Add legend
axes.iloc[-1, 0].legend(
    bbox_to_anchor=(len(axes)/2, len(axes)), 
    loc='lower center', 
    ncols=2
)
```

**Parameter Names**: To define parameters for plots, create a list of parameter names and pass them to both `read_chains()` (via `columns` parameter) and the plotting functions.

### Example 3: Computing Statistics

```python
samples = read_chains("../../tests/example_data/pc")

# Compute nested sampling statistics
stats = samples.stats(nsamples=2000)

# Access evidence and other quantities
print(f"log(Z) = {stats['logZ']}")
print(f"D_KL = {stats['D']}")
```

## Key Methods

- `read_chains()`: Load sampling data from various formats
- `make_1d_axes()`: Create axes for 1D marginal plots
- `make_2d_axes()`: Create axes for 2D marginal plots (triangle plots)
- `samples.plot_1d()`: Plot 1D marginal posteriors
- `samples.plot_2d()`: Plot 2D marginal posteriors
- `samples.prior()`: Extract prior samples
- `samples.stats()`: Compute nested sampling statistics

## Integration with Other Tools

Anesthetic works seamlessly with:
- NumPy for numerical operations
- Matplotlib for plotting
- Pandas for data manipulation
- Other nested sampling packages (PolyChord, MultiNest, dynesty)

## Nested Sampling Chain File Format

### Dead-Birth Format

The dead-birth format is used to store nested sampling chains in a way that allows reconstruction of the full run and enables dynamic nested sampling. Each row represents a point in parameter space along with its likelihood information.

#### File Structure

The file contains `ndims + 2` columns in space-separated format:
```
param1 param2 ... paramN logL logL_birth
```

where:
- `param1` to `paramN`: Parameter values
- `logL`: Log-likelihood value at death (when point was discarded)
- `logL_birth`: Log-likelihood value at birth (contour where point was sampled)

#### Example

For a 2D Gaussian:
```
0.523 -1.234  -45.67  -50.12
1.456  0.789  -44.32  -48.89
-0.234 0.567  -43.21  -47.65
```

### Saving Chains in Dead-Birth Format

```python
import numpy as np

def save_ns_chains(points, logL_death, logL_birth, filename='chains_dead-birth.txt'):
    """Save nested sampling chains in dead-birth format.
    
    Args:
        points: Array of shape (n_samples, n_dims) - parameter values
        logL_death: Array of shape (n_samples,) - death likelihood values
        logL_birth: Array of shape (n_samples,) - birth likelihood values
        filename: Output filename
    """
    # Stack data: parameters, death likelihood, birth likelihood
    data = np.column_stack([points, logL_death, logL_birth])
    # Save without headers (space-separated)
    np.savetxt(filename, data)

# Example usage with blackjax NSInfo object:
def save_from_nsinfo(dead_info, filename='chains_dead-birth.txt'):
    points = np.array(dead_info.particles)
    logL_death = np.array(dead_info.logL)
    logL_birth = np.array(dead_info.logL_birth)
    save_ns_chains(points, logL_death, logL_birth, filename)
```

### Key Points

1. No headers in the file
2. Space-separated values
3. Each row is: parameters + death logL + birth logL
4. Birth contours enable reconstruction of NS run
5. Format compatible with analysis tools like anesthetic
6. File naming convention: `[root]_dead-birth.txt`