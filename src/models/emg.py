"""
Exponential Modified Gaussian (EMG) model for FRB pulses
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict
from . import BaseModel, ModelConfig


class EMGModel(BaseModel):
    """Exponential Modified Gaussian model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def model_function(self, t: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute EMG model prediction
        
        Args:
            t: Time array
            params: Dictionary with parameters A, tau, u, w, sigma, and optionally Npulse
            
        Returns:
            Model prediction at time points t
        """
        A = params['A']
        tau = params['tau']
        u = params['u']
        w = params['w']
        
        # If fitting pulses, use Npulse to determine how many peaks to include
        if self.fit_pulses and 'Npulse' in params:
            # Keep Npulse continuous, only use for comparison
            Npulse = jnp.minimum(params['Npulse'][0], self.max_peaks)
        else:
            Npulse = self.max_peaks
        
        # Initialize output
        pp = jnp.zeros_like(t)
        
        # Add each pulse contribution
        for i in range(self.max_peaks):
            # Use continuous comparison for smooth gradients
            weight = jax.nn.sigmoid(10.0 * (Npulse - i - 0.5))
            
            # EMG function
            sqrt2 = jnp.sqrt(2)
            exp_arg = (w[i]**2) / (2 * tau[i]**2) - (t - u[i]) / tau[i]
            erf_arg = ((t - u[i]) / (sqrt2 * w[i])) - w[i] / (sqrt2 * tau[i])
            
            # Compute EMG with numerical stability
            exp_term = jnp.exp(jnp.minimum(exp_arg, 20.0))  # Prevent overflow
            erf_term = 0.5 * (1 + jax.scipy.special.erf(erf_arg))
            
            pulse = A[i] * exp_term * erf_term
            pp += weight * pulse
        
        return pp
    
    def sample_prior(self, key: jax.random.PRNGKey, prior_config: Dict) -> Dict[str, jnp.ndarray]:
        """Sample parameters from prior distribution"""
        params = {}
        
        # Sample amplitude (uniform)
        A_range = prior_config.get('amplitude', {'min': 0.001, 'max': 0.1})
        key, subkey = random.split(key)
        params['A'] = random.uniform(
            subkey,
            shape=(self.max_peaks,),
            minval=A_range['min'],
            maxval=A_range['max']
        )
        
        # Sample tau (uniform)
        tau_range = prior_config.get('tau', {'min': 0.1, 'max': 1.0})
        key, subkey = random.split(key)
        params['tau'] = random.uniform(
            subkey,
            shape=(self.max_peaks,),
            minval=tau_range['min'],
            maxval=tau_range['max']
        )
        
        # Sample u (uniform)
        u_range = prior_config.get('u', {'min': 0.0, 'max': 5.0})
        key, subkey = random.split(key)
        params['u'] = random.uniform(
            subkey,
            shape=(self.max_peaks,),
            minval=u_range['min'],
            maxval=u_range['max']
        )
        
        # Sample width (uniform)
        w_range = prior_config.get('width', {'min': 0.05, 'max': 0.5})
        key, subkey = random.split(key)
        params['w'] = random.uniform(
            subkey,
            shape=(self.max_peaks,),
            minval=w_range['min'],
            maxval=w_range['max']
        )
        
        # Sample sigma (log-uniform)
        sigma_range = prior_config.get('sigma', {'min': 1e-4, 'max': 0.01})
        key, subkey = random.split(key)
        log_sigma = random.uniform(
            subkey,
            shape=(1,),
            minval=jnp.log(sigma_range['min']),
            maxval=jnp.log(sigma_range['max'])
        )
        params['sigma'] = jnp.exp(log_sigma)
        
        # Sample Npulse if fit_pulses is True (continuous)
        if self.fit_pulses:
            Npulse_range = prior_config.get('Npulse', {'min': 1, 'max': self.max_peaks})
            key, subkey = random.split(key)
            params['Npulse'] = random.uniform(
                subkey,
                shape=(1,),
                minval=Npulse_range['min'],
                maxval=Npulse_range['max']
            )
            
            # Only cast to int when needed for array operations
            Npulse_int = jnp.minimum(params['Npulse'][0].astype(int), self.max_peaks)
            
            # Zero out parameters beyond Npulse_int for initial guess
            for i in range(self.max_peaks):
                mask = (i < Npulse_int).astype(float)
                params['A'] = params['A'].at[i].multiply(mask)
        
        return params