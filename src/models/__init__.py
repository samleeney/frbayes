"""
Models for FRBayes JAX implementation
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import jax.numpy as jnp


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    max_peaks: int = 3
    fit_pulses: bool = False
    model_type: str = 'emg'
    
    @property
    def ndims(self) -> int:
        """Calculate number of dimensions"""
        base_dims = self.max_peaks * 4 + 1  # A, tau, u, w + sigma
        if self.fit_pulses:
            base_dims += 1  # Add Npulse
        return base_dims


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_peaks = config.max_peaks
        self.fit_pulses = config.fit_pulses
        self.ndims = config.ndims
    
    def model_function(self, t: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute model prediction"""
        raise NotImplementedError
    
    def sample_prior(self, key, prior_config: Dict) -> Dict[str, jnp.ndarray]:
        """Sample from prior distribution"""
        raise NotImplementedError


def get_model(model_type: str, config: ModelConfig) -> BaseModel:
    """Factory function to get model instance"""
    if model_type == 'emg':
        from .emg import EMGModel
        return EMGModel(config)
    elif model_type == 'exponential':
        from .exponential import ExponentialModel
        return ExponentialModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")