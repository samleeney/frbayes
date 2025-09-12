"""
Simple prior distributions for FRB models using distrax.
"""
import jax
import jax.numpy as jnp
import distrax
from typing import Dict, Optional


def forced_identifiability_transform(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform uniform samples to maintain sorted order.
    Based on the transform from the original frbayes implementation.
    
    Args:
        x: Array of uniform samples in [0, 1]
    
    Returns:
        Transformed samples that maintain ordering
    """
    N = len(x)
    t = jnp.zeros_like(x)
    
    # Use JAX's functional approach for the transform
    # Start from the last element
    t = t.at[N-1].set(x[N-1]**(1./N))
    
    # Work backwards using a scan or explicit loop
    for n in range(N-2, -1, -1):
        t = t.at[n].set(x[n]**(1./(n+1)) * t[n+1])
    
    return t


class FRBPriors:
    """
    Simple prior system using distrax distributions.
    """
    
    def __init__(
        self, 
        model_name: str, 
        max_peaks: int, 
        fit_pulses: bool,
        prior_bounds: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.max_peaks = max_peaks
        self.fit_pulses = fit_pulses
        
        # Set default bounds if not provided
        if prior_bounds is None:
            prior_bounds = {
                'amplitude': {'min': 0.001, 'max': 10.0},
                'tau': {'min': 0.001, 'max': 10.0},
                'u': {'min': 0.0, 'max': 4.0},  # Should be set to actual time range
                'width': {'min': 0.001, 'max': 5.0},
                'baseline': {'min': -1.0, 'max': 1.0},
                'log_sigma': {'min': jnp.log(0.0001), 'max': jnp.log(2.0)},
                'spectral_index': {'min': -3.0, 'max': 1.0},  # Typical range for FRBs
            }
        self.prior_bounds = prior_bounds
        
        # Calculate number of dimensions
        if self.model_name == "emg":
            self.ndims = 4 * max_peaks + 1  # A, tau, u, w for each peak + sigma
        elif self.model_name == "exponential":
            self.ndims = 3 * max_peaks + 1  # A, tau, u for each peak + sigma
        elif self.model_name == "emg_with_baseline":
            self.ndims = 4 * max_peaks + 2  # A, tau, u, w for each peak + baseline + sigma
        elif self.model_name == "exponential_with_baseline":
            self.ndims = 3 * max_peaks + 2  # A, tau, u for each peak + baseline + sigma
        elif self.model_name == "periodic_exponential":
            self.ndims = 2 * max_peaks + 3  # A, tau for each peak + u0 + period + sigma
        elif self.model_name == "periodic_exponential_with_baseline":
            self.ndims = 2 * max_peaks + 4  # A, tau for each peak + u0 + period + baseline + sigma
        # 2D models with spectral index
        elif self.model_name == "emg_2d":
            self.ndims = 4 * max_peaks + 2  # A, tau, u, w for each peak + alpha + sigma
        elif self.model_name == "exponential_2d":
            self.ndims = 3 * max_peaks + 2  # A, tau, u for each peak + alpha + sigma
        elif self.model_name == "emg_2d_with_baseline":
            self.ndims = 4 * max_peaks + 3  # A, tau, u, w for each peak + baseline + alpha + sigma
        elif self.model_name == "exponential_2d_with_baseline":
            self.ndims = 3 * max_peaks + 3  # A, tau, u for each peak + baseline + alpha + sigma
        else:
            raise ValueError(f"Model {self.model_name} not recognized")
        
        if fit_pulses:
            self.ndims += 1
    
    def sample_from_prior(self, rng_key: jax.random.PRNGKey, n_samples: int = 1) -> jnp.ndarray:
        """
        Sample from uniform/log-uniform priors directly.
        """
        samples = []
        keys = jax.random.split(rng_key, self.ndims)
        key_idx = 0
        
        # Amplitudes - uniform
        for i in range(self.max_peaks):
            dist = distrax.Uniform(
                low=self.prior_bounds['amplitude']['min'],
                high=self.prior_bounds['amplitude']['max']
            )
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
            key_idx += 1
        
        # Taus - uniform
        for i in range(self.max_peaks):
            dist = distrax.Uniform(
                low=self.prior_bounds['tau']['min'],
                high=self.prior_bounds['tau']['max']
            )
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
            key_idx += 1
        
        # Arrival times/location parameters
        if 'periodic' in self.model_name:
            # For periodic models: u0 and period
            # u0 - first pulse location
            u0_bounds = self.prior_bounds.get('u0', self.prior_bounds.get('u', {'min': -5.0, 'max': 10.0}))
            dist = distrax.Uniform(
                low=u0_bounds['min'],
                high=u0_bounds['max']
            )
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
            key_idx += 1
            
            # period - spacing between pulses
            dist = distrax.Uniform(
                low=self.prior_bounds['period']['min'],
                high=self.prior_bounds['period']['max']
            )
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
            key_idx += 1
        else:
            # Non-periodic models: individual arrival times (sorted)
            if self.max_peaks > 1:
                # Sample uniform [0, 1] values
                u_samples_01 = []
                for i in range(self.max_peaks):
                    dist = distrax.Uniform(low=0.0, high=1.0)
                    u_samples_01.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
                    key_idx += 1
                
                # Stack and apply transform to each sample
                u_01_stacked = jnp.stack(u_samples_01, axis=-1)  # Shape: (n_samples, max_peaks)
                
                # Apply transform to each sample using vmap
                u_sorted_01 = jax.vmap(forced_identifiability_transform)(u_01_stacked)
                
                # Rescale to [u_min, u_max]
                u_min = self.prior_bounds['u']['min']
                u_max = self.prior_bounds['u']['max']
                u_sorted = u_min + (u_max - u_min) * u_sorted_01
                
                # Split back into individual samples for consistency with rest of code
                for i in range(self.max_peaks):
                    samples.append(u_sorted[:, i])
            else:
                # Single peak - no sorting needed
                dist = distrax.Uniform(
                    low=self.prior_bounds['u']['min'],
                    high=self.prior_bounds['u']['max']
                )
                samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
                key_idx += 1
        
        # Widths (for EMG) - uniform
        if 'emg' in self.model_name:
            for i in range(self.max_peaks):
                dist = distrax.Uniform(
                    low=self.prior_bounds['width']['min'],
                    high=self.prior_bounds['width']['max']
                )
                samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
                key_idx += 1
        
        # Baseline (if applicable) - uniform
        if 'baseline' in self.model_name:
            dist = distrax.Uniform(
                low=self.prior_bounds['baseline']['min'],
                high=self.prior_bounds['baseline']['max']
            )
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
            key_idx += 1
        
        # Spectral index (for 2D models) - uniform
        if '2d' in self.model_name:
            dist = distrax.Uniform(
                low=self.prior_bounds['spectral_index']['min'],
                high=self.prior_bounds['spectral_index']['max']
            )
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
            key_idx += 1
        
        # Sigma - log-uniform (sample in log space, then exp)
        log_dist = distrax.Uniform(
            low=self.prior_bounds['log_sigma']['min'],
            high=self.prior_bounds['log_sigma']['max']
        )
        log_sigma = log_dist.sample(seed=keys[key_idx], sample_shape=(n_samples,))
        samples.append(jnp.exp(log_sigma))
        key_idx += 1
        
        # Npulse (if fitted) - uniform integer
        if self.fit_pulses:
            dist = distrax.Uniform(low=1.0, high=float(self.max_peaks))
            samples.append(dist.sample(seed=keys[key_idx], sample_shape=(n_samples,)))
        
        return jnp.stack(samples, axis=-1)