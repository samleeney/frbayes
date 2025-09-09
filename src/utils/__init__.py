"""
Utility functions for FRBayes JAX implementation
"""

import os
import jax


def configure_gpu(memory_fraction: float = 0.8, force_cpu: bool = False):
    """
    Configure GPU memory usage for JAX
    
    Args:
        memory_fraction: Fraction of GPU memory to allocate
        force_cpu: If True, force CPU usage even if GPU is available
    """
    if force_cpu:
        jax.config.update('jax_platform_name', 'cpu')
        print("Forced CPU usage")
        return
    
    # Set GPU memory growth
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
    
    # Check available devices
    devices = jax.devices()
    if any('gpu' in str(d).lower() for d in devices):
        print(f"GPU detected. Configured with {memory_fraction:.0%} memory allocation")
    else:
        print("No GPU detected. Using CPU")
    
    return devices