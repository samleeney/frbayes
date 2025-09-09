"""
Exponential decay model
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict
from . import BaseModel, ModelConfig


class ExponentialModel(BaseModel):
    """Simple exponential decay model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    def model_function(self, t: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute exponential model prediction
        
        Args:
            t: Time array
            params: Dictionary with parameters A, tau, u, and optionally Npulse
            
        Returns:
            Model prediction at time points t
        """
        A = params['A']
        tau = params['tau']
        u = params['u']
        
        # If fitting pulses, use Npulse to determine how many peaks to include
        if self.fit_pulses and 'Npulse' in params:
            Npulse = jnp.minimum(params['Npulse'][0], self.max_peaks)
        else:
            Npulse = self.max_peaks
        
        # Initialize output
        pp = jnp.zeros_like(t)
        
        # Add each pulse contribution
        for i in range(self.max_peaks):
            # Use continuous comparison for smooth gradients
            weight = jax.nn.sigmoid(10.0 * (Npulse - i - 0.5))
            
            # Exponential function with step at u[i]
            pulse = A[i] * jnp.exp(-(t - u[i]) / tau[i]) * (t >= u[i])
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
        
        # No width parameter for exponential model
        # But we need to include it for compatibility
        params['w'] = jnp.ones(self.max_peaks) * 0.1
        
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
        
        return params