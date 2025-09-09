"""
Utility functions for FRBayes JAX.
"""
import yaml
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any


def load_settings(filename: str = "settings.yaml") -> Dict[str, Any]:
    """
    Load settings from YAML file.
    
    Args:
        filename: Path to settings file
    
    Returns:
        Dictionary of settings
    """
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def get_default_prior_ranges(model_name: str) -> Dict:
    """
    Get default prior ranges for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary of prior ranges
    """
    # Common priors
    common = {
        "amplitude": {"min": 0.0001, "max": 15},
        "tau": {"min": 0.1, "max": 1},
        "u": {"min": 0.01, "max": 4.0},
        "sigma": {"min": 0.00001, "max": 0.1}
    }
    
    # Model-specific additions
    if "emg" in model_name:
        common["width"] = {"min": 0.001, "max": 0.3}
    
    if "baseline" in model_name:
        common["baseline_offset"] = {"min": -1.0, "max": 1.0}
    
    # Exponential models use different amplitude range
    if "exponential" in model_name and "emg" not in model_name:
        common["amplitude"] = {"min": 0.001, "max": 0.1}
    
    return common


def extract_prior_ranges_from_settings(settings: Dict, model_name: str) -> Dict:
    """
    Extract prior ranges from settings for a specific model.
    
    Args:
        settings: Settings dictionary
        model_name: Name of the model
    
    Returns:
        Dictionary of prior ranges
    """
    # Start with defaults
    prior_ranges = get_default_prior_ranges(model_name)
    
    # Override with settings if available
    if "prior_ranges" in settings:
        pr = settings["prior_ranges"]
        
        # Common priors
        for key in ["amplitude", "tau", "u", "sigma"]:
            if key in pr:
                prior_ranges[key] = pr[key]
        
        # Model-specific priors
        if model_name in pr:
            model_pr = pr[model_name]
            for key, value in model_pr.items():
                prior_ranges[key] = value
        
        # Handle baseline models
        if "baseline" in model_name:
            base_model = model_name.replace("_with_baseline", "")
            if f"{base_model}_with_baseline" in pr:
                baseline_pr = pr[f"{base_model}_with_baseline"]
                if "baseline_offset" in baseline_pr:
                    prior_ranges["baseline_offset"] = baseline_pr["baseline_offset"]
    
    return prior_ranges