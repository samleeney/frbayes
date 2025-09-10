"""
Utility functions for FRBayes JAX.
"""
import numpy as np
import jax.numpy as jnp


def fix_nan_logL_birth(final_state):
    """
    Fix NaN values in logL_birth from BlackJAX's finalise function.
    
    BlackJAX's finalise function leaves NaN values for the birth likelihoods
    of the final live points (those that survived from initialization to the end).
    This breaks anesthetic's weight calculations.
    
    Following standard nested sampling practice (dynesty, PolyChord), we set
    the birth likelihood of final live points to the maximum of the existing
    birth likelihoods, since these points survived to the very end and represent
    the highest likelihood region.
    
    Args:
        final_state: The finalized state from BlackJAX nested sampling
        
    Returns:
        Fixed logL_birth array with no NaN values
    """
    logL_birth = np.array(final_state.loglikelihood_birth)
    nan_mask = np.isnan(logL_birth)
    
    if np.any(nan_mask):
        # Standard approach: final live points get the maximum birth likelihood
        # since they survived to the end (highest likelihood region)
        max_valid = np.nanmax(logL_birth)
        logL_birth[nan_mask] = max_valid
    
    return logL_birth