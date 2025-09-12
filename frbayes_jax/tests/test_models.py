"""
Test all models with simulated 3-pulse data using the 'fitted' option.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling


class TestModels:
    """Test suite for all models with 3 pulses."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up common test parameters."""
        self.max_peaks = 4  # Allow up to 4 peaks to test model selection
        self.fit_pulses = True  # Fit number of pulses
        self.true_npulse = 3
        self.tolerance = 0.05  # 5% tolerance
        
        # Common time array
        self.t = jnp.linspace(0.0, 5.0, 500)
        self.t_np = np.array(self.t)
        
        # Common nested sampling parameters
        self.ns_params = {
            'num_live_points': 200,
            'num_delete': 100,
            'num_inner_steps': 50,
            'log_tolerance': -2.0,
            'seed': 42
        }
    
    def run_model_test(self, model_name, true_params, true_npulse_for_sim, prior_bounds):
        """Helper function to run a model test."""
        # Get model function
        model_func = get_model_function(model_name)
        
        # Simulate data
        _, data = simulate_frb_data(
            model_func, true_params, true_npulse_for_sim, False,
            t_min=0.0, t_max=5.0, num_points=500,
            add_noise=True, seed=42
        )
        data_np = np.array(data)
        
        # Run nested sampling
        final_state = run_nested_sampling(
            model_name=model_name,
            data=data_np,
            t=self.t_np,
            prior_bounds=prior_bounds,
            max_peaks=self.max_peaks,
            fit_pulses=self.fit_pulses,
            **self.ns_params
        )
        
        # Get parameter names and extract results
        param_names = get_param_names(model_name, self.max_peaks, self.fit_pulses)
        
        # Calculate posterior mean
        weights = np.exp(final_state.loglikelihood - np.max(final_state.loglikelihood))
        weights /= np.sum(weights)
        best_fit = np.average(final_state.particles, weights=weights, axis=0)
        
        # Extract fitted number of pulses
        fitted_npulse = best_fit[-1] if self.fit_pulses else self.max_peaks
        
        # Check if fitted number of pulses is close to true value
        assert abs(fitted_npulse - self.true_npulse) <= 1.0, \
            f"Model {model_name}: Fitted Npulse={fitted_npulse:.1f}, expected {self.true_npulse}"
        
        # Check key parameters are within tolerance
        # We'll check the first few amplitudes and sigma
        for i in range(min(3, true_npulse_for_sim)):
            fitted_amp = best_fit[i]
            true_amp = true_params[i]
            relative_error = abs(fitted_amp - true_amp) / true_amp
            assert relative_error < 0.5, \
                f"Model {model_name}: Amplitude {i+1} error {relative_error:.2%} exceeds 50%"
        
        return fitted_npulse, best_fit
    
    def test_emg_model(self):
        """Test EMG model with 3 pulses."""
        # True parameters for 3 EMG pulses
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            1.0, 2.0, 3.5,    # Arrival times
            0.2, 0.15, 0.18,  # Width parameters
            0.05              # Sigma
        ])
        
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'width': {'min': 0.05, 'max': 0.5},  # Changed from 'w' to 'width'
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        fitted_npulse, best_fit = self.run_model_test('emg', true_params, 3, prior_bounds)
        print(f"EMG model: Fitted Npulse={fitted_npulse:.1f}")
    
    def test_exponential_model(self):
        """Test exponential model with 3 pulses."""
        # True parameters for 3 exponential pulses
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            1.0, 2.0, 3.5,    # Arrival times
            0.05              # Sigma
        ])
        
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        fitted_npulse, best_fit = self.run_model_test('exponential', true_params, 3, prior_bounds)
        print(f"Exponential model: Fitted Npulse={fitted_npulse:.1f}")
    
    def test_emg_with_baseline_model(self):
        """Test EMG with baseline model with 3 pulses."""
        # True parameters for 3 EMG pulses with baseline
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            1.0, 2.0, 3.5,    # Arrival times
            0.2, 0.15, 0.18,  # Width parameters
            0.1,              # Baseline offset
            0.05              # Sigma
        ])
        
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'width': {'min': 0.05, 'max': 0.5},  # Changed from 'w' to 'width'
            'baseline': {'min': -0.2, 'max': 0.3},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        fitted_npulse, best_fit = self.run_model_test('emg_with_baseline', true_params, 3, prior_bounds)
        print(f"EMG with baseline model: Fitted Npulse={fitted_npulse:.1f}")
    
    def test_exponential_with_baseline_model(self):
        """Test exponential with baseline model with 3 pulses."""
        # True parameters for 3 exponential pulses with baseline
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            1.0, 2.0, 3.5,    # Arrival times
            0.1,              # Baseline offset
            0.05              # Sigma
        ])
        
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'baseline': {'min': -0.2, 'max': 0.3},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        fitted_npulse, best_fit = self.run_model_test('exponential_with_baseline', true_params, 3, prior_bounds)
        print(f"Exponential with baseline model: Fitted Npulse={fitted_npulse:.1f}")
    
    def test_periodic_exponential_model(self):
        """Test periodic exponential model with 3 pulses."""
        # True parameters for 3 periodic exponential pulses
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            0.5,              # u0 (first pulse location)
            1.3,              # period
            0.05              # Sigma
        ])
        
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u0': {'min': 0.0, 'max': 2.0},
            'period': {'min': 0.5, 'max': 2.0},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        fitted_npulse, best_fit = self.run_model_test('periodic_exponential', true_params, 3, prior_bounds)
        print(f"Periodic exponential model: Fitted Npulse={fitted_npulse:.1f}")
    
    def test_periodic_exponential_with_baseline_model(self):
        """Test periodic exponential with baseline model with 3 pulses."""
        # True parameters for 3 periodic exponential pulses with baseline
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            0.5,              # u0 (first pulse location)
            1.3,              # period
            0.1,              # Baseline offset
            0.05              # Sigma
        ])
        
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u0': {'min': 0.0, 'max': 2.0},
            'period': {'min': 0.5, 'max': 2.0},
            'baseline': {'min': -0.2, 'max': 0.3},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        fitted_npulse, best_fit = self.run_model_test('periodic_exponential_with_baseline', true_params, 3, prior_bounds)
        print(f"Periodic exponential with baseline model: Fitted Npulse={fitted_npulse:.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])