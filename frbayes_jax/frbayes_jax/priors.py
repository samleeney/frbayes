"""
Simple prior distributions for FRB models using distrax.
"""
import jax
import jax.numpy as jnp
import distrax
from typing import Dict, Optional


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
                'u': {'min': -5.0, 'max': 10.0},
                'width': {'min': 0.001, 'max': 5.0},
                'baseline': {'min': -1.0, 'max': 1.0},
                'log_sigma': {'min': jnp.log(0.0001), 'max': jnp.log(2.0)},
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
        
        # Arrival times - uniform
        for i in range(self.max_peaks):
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