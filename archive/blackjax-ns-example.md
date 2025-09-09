# BlackJAX Nested Sampling

A JAX-based implementation of nested sampling algorithms, providing composable and vectorized components for Bayesian inference.

## Overview

BlackJAX nested sampling is a fork of the main BlackJAX repository that implements nested sampling algorithms in JAX. It provides "atomic components" of nested sampling with clear separation of design choices from the core algorithm, enabling advanced experimentation and native vectorization.

## Installation

Install from the forked repository:

```bash
pip install git+https://github.com/handley-lab/blackjax@nested_sampling
```

**Note**: This is not yet part of the main BlackJAX repository. The implementation is under active development.

## Documentation

- [Nested Sampling Book](https://handley-lab.co.uk/nested-sampling-book/intro.html)
- [BlackJAX Repository](https://github.com/handley-lab/blackjax/tree/nested_sampling)

## Key Features

- Fully JAX-compatible for automatic differentiation and JIT compilation
- Composable components for algorithm experimentation
- Natively vectorized likelihood code
- Integration with modern probabilistic programming libraries (NumPyro)
- Efficient particle Monte Carlo implementation

## Quickstart Example

```python
import jax
import jax.numpy as jnp
import blackjax
import tqdm 
from blackjax.ns.utils import finalise

# Set random seed
rng_key = jax.random.PRNGKey(0)

# Define likelihood and prior
loglikelihood_function = lambda x: jax.scipy.stats.multivariate_normal.logpdf(
    x, jnp.ones(5), jnp.eye(5)*0.01
)
logprior_function = lambda x: jax.scipy.stats.norm.logpdf(x).sum()

# Initialize nested sampling algorithm
algo = blackjax.nss(
    logprior_fn=logprior_function,
    loglikelihood_fn=loglikelihood_function,
    num_delete=50,           # Number of points to delete per iteration
    num_inner_steps=20,      # Number of MCMC steps between replacements
)

# Initialize state with live points
rng_key, sampling_key, initialization_key = jax.random.split(rng_key, 3)
initial_live_points = jax.random.normal(initialization_key, (1000, 5))
state = algo.init(initial_live_points)

# JIT-compile the step function
@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), dead_point

# Run nested sampling with progress bar
dead = []

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while (not state.logZ_live - state.logZ < -3):  # Termination criterion
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(50)

# Finalize results
final_state = finalise(state, dead)
```

## Integration with Anesthetic

Convert results to anesthetic format for visualization:

```python
import anesthetic
import numpy as np

# Create NestedSamples object
nested_samples = anesthetic.NestedSamples(
    data=final_state.particles,
    logL=final_state.loglikelihood,
    logL_birth=final_state.loglikelihood_birth,
)

# Plot prior and posterior
prior = nested_samples.set_beta(0.0).plot_2d(np.arange(5), label="prior")
post = nested_samples.plot_2d(prior, label="posterior")
prior.iloc[-1, 0].legend(
    bbox_to_anchor=(len(prior), len(prior)), 
    loc='lower right'
)
```

## Key Parameters

### Algorithm Initialization (`blackjax.nss`)
- `logprior_fn`: Function computing log prior probability
- `loglikelihood_fn`: Function computing log likelihood
- `num_delete`: Number of live points to replace per iteration
- `num_inner_steps`: Number of MCMC steps for generating new points

### State Components
- `state.logZ`: Current log evidence estimate
- `state.logZ_live`: Log evidence of remaining live points
- `state.particles`: Current live point positions
- `state.loglikelihood`: Log likelihood values

## Advanced Usage

### Custom Termination Criteria

```python
# Continue until evidence uncertainty is small
while state.logZ_live - state.logZ > log_tolerance:
    # ... sampling step ...
```

### Parallel Chains

```python
# Initialize multiple independent chains
vmapped_init = jax.vmap(algo.init)
multiple_states = vmapped_init(initial_points_batch)
```

## Citation

When using this implementation, please cite:

1. BlackJAX repository: Cabezas et al. (2024)
2. Implementation paper: Yallup and Handley (2025) [pending]

## Notes

- The implementation leverages JAX's JIT compilation for performance
- All operations are differentiable, enabling gradient-based optimization
- Compatible with JAX transformations (vmap, pmap, etc.)
- Designed for both research and production use cases