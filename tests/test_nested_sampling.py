#!/usr/bin/env python3
"""
Test suite for nested sampling implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from models import get_model, ModelConfig
from inference.blackjax_nested_sampling import BlackJAXNestedSampler
from utils import configure_gpu


@pytest.fixture
def setup_gpu():
    """Configure GPU for tests"""
    configure_gpu(memory_fraction=0.8, force_cpu=False)


@pytest.fixture
def test_data():
    """Generate test data with 2 peaks"""
    t = jnp.linspace(0, 5, 100)
    true_params = {
        'A': jnp.array([0.08, 0.06]),
        'tau': jnp.array([0.5, 0.3]),
        'u': jnp.array([1.5, 3.0]),
        'w': jnp.array([0.2, 0.15]),
        'sigma': jnp.array([0.003])
    }
    
    # Generate noisy data
    key = jax.random.PRNGKey(42)
    model = get_model('emg', ModelConfig(max_peaks=2, fit_pulses=False))
    pp = model.model_function(t, true_params)
    pp += true_params['sigma'][0] * jax.random.normal(jax.random.split(key)[1], pp.shape)
    
    return {'pp': pp, 't': t}, true_params


def test_2peak_fixed(setup_gpu, test_data):
    """Test 2-peak model with fixed number of peaks"""
    data, true_params = test_data
    
    # Create model
    model = get_model('emg', ModelConfig(max_peaks=2, fit_pulses=False, model_type='emg'))
    
    # Set up sampler
    ndims = model.ndims
    nlive = ndims * 25
    
    sampler = BlackJAXNestedSampler(
        model, data,
        prior_config={
            'amplitude': {'min': 0.001, 'max': 0.2},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'width': {'min': 0.05, 'max': 0.5},
            'sigma': {'min': 1e-4, 'max': 0.01}
        }
    )
    
    # Run sampling
    key = jax.random.PRNGKey(42)
    results = sampler.run(key, nlive, 50000, 3.0)
    
    # Check results
    assert 'samples' in results
    assert 'log_evidence' in results
    assert len(results['samples']) > 0
    
    # Check parameter recovery (loose bounds for test)
    posterior_means = {}
    for i, sample in enumerate(results['samples'][:100]):  # Use first 100 samples
        for key in ['A', 'tau', 'u', 'w', 'sigma']:
            if key not in posterior_means:
                posterior_means[key] = []
            posterior_means[key].append(sample[key])
    
    # Calculate means
    for key in posterior_means:
        posterior_means[key] = np.mean(posterior_means[key], axis=0)
    
    # Check recovery within reasonable bounds
    assert np.abs(posterior_means['A'][0] - true_params['A'][0]) < 0.05
    assert np.abs(posterior_means['u'][0] - true_params['u'][0]) < 0.5


def test_2peak_fitpulses(setup_gpu, test_data):
    """Test 2-peak model with variable number of peaks"""
    data, true_params = test_data
    
    # Create model with fit_pulses=True
    model = get_model('emg', ModelConfig(max_peaks=3, fit_pulses=True, model_type='emg'))
    
    # Extend true_params for 3 peaks (3rd peak has 0 amplitude)
    true_params_extended = {
        'A': jnp.array([0.08, 0.06, 0.0]),
        'tau': jnp.array([0.5, 0.3, 0.1]),
        'u': jnp.array([1.5, 3.0, 4.5]),
        'w': jnp.array([0.2, 0.15, 0.1]),
        'sigma': jnp.array([0.003]),
        'Npulse': jnp.array([2.0])
    }
    
    # Set up sampler
    ndims = model.ndims
    nlive = ndims * 25
    
    sampler = BlackJAXNestedSampler(
        model, data,
        prior_config={
            'amplitude': {'min': 0.001, 'max': 0.2},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'width': {'min': 0.05, 'max': 0.5},
            'sigma': {'min': 1e-4, 'max': 0.01},
            'Npulse': {'min': 1, 'max': 3}
        }
    )
    
    # Run sampling
    key = jax.random.PRNGKey(42)
    results = sampler.run(key, nlive, 50000, 3.0)
    
    # Check results
    assert 'samples' in results
    assert 'log_evidence' in results
    assert len(results['samples']) > 0
    
    # Check Npulse recovery
    npulse_samples = [s['Npulse'][0] for s in results['samples'][:100]]
    npulse_mean = np.mean(npulse_samples)
    
    # Should favor 2 peaks
    assert 1.5 < npulse_mean < 2.5


def test_convergence_criterion(setup_gpu):
    """Test that sampling stops based on convergence criterion"""
    # Simple 1-peak model for faster convergence
    model = get_model('emg', ModelConfig(max_peaks=1, fit_pulses=False))
    
    # Generate simple data
    t = jnp.linspace(0, 5, 50)
    true_params = {
        'A': jnp.array([0.08]),
        'tau': jnp.array([0.5]),
        'u': jnp.array([2.5]),
        'w': jnp.array([0.2]),
        'sigma': jnp.array([0.001])
    }
    
    key = jax.random.PRNGKey(42)
    pp = model.model_function(t, true_params)
    pp += true_params['sigma'][0] * jax.random.normal(jax.random.split(key)[1], pp.shape)
    data = {'pp': pp, 't': t}
    
    # Set up sampler with tight convergence
    sampler = BlackJAXNestedSampler(
        model, data,
        prior_config={
            'amplitude': {'min': 0.05, 'max': 0.12},
            'tau': {'min': 0.3, 'max': 0.7},
            'u': {'min': 2.0, 'max': 3.0},
            'width': {'min': 0.1, 'max': 0.3},
            'sigma': {'min': 5e-4, 'max': 0.002}
        }
    )
    
    # Run with different precision criteria
    results_loose = sampler.run(key, 200, 50000, 1.0)  # Loose criterion
    results_tight = sampler.run(key, 200, 50000, 5.0)  # Tight criterion
    
    # Tight criterion should run more iterations
    assert results_tight['num_iterations'] >= results_loose['num_iterations']