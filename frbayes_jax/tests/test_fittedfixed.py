"""
Test fitted vs fixed number of pulses with 3 exponential pulses.
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


class TestFittedFixed:
    """Test suite for comparing fitted vs fixed pulse number estimation."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up common test parameters."""
        self.model_name = "exponential"
        self.true_npulse = 3
        
        # True parameters for 3 exponential pulses
        self.true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            1.0, 2.0, 3.5,    # Arrival times
            0.05              # Sigma
        ])
        
        # Prior bounds
        self.prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        # Time array
        self.t = jnp.linspace(0.0, 5.0, 500)
        self.t_np = np.array(self.t)
        
        # Simulate data once for both tests
        model_func = get_model_function(self.model_name)
        _, self.data = simulate_frb_data(
            model_func, self.true_params, self.true_npulse, False,
            t_min=0.0, t_max=5.0, num_points=500,
            add_noise=True, seed=42
        )
        self.data_np = np.array(self.data)
        
        # Common nested sampling parameters (reduced for faster testing)
        self.ns_params = {
            'num_live_points': 150,
            'num_delete': 75,
            'num_inner_steps': 40,
            'log_tolerance': -2.0,
            'seed': 123
        }
    
    def test_fitted_pulses(self):
        """Test with fitted number of pulses."""
        max_peaks = 4  # Allow up to 4 peaks
        fit_pulses = True
        
        print("\nTesting FITTED pulse number estimation...")
        print(f"True number of pulses: {self.true_npulse}")
        print(f"Max peaks allowed: {max_peaks}")
        
        # Run nested sampling
        final_state = run_nested_sampling(
            model_name=self.model_name,
            data=self.data_np,
            t=self.t_np,
            prior_bounds=self.prior_bounds,
            max_peaks=max_peaks,
            fit_pulses=fit_pulses,
            **self.ns_params
        )
        
        # Get parameter names
        param_names = get_param_names(self.model_name, max_peaks, fit_pulses)
        
        # Calculate posterior mean
        weights = np.exp(final_state.loglikelihood - np.max(final_state.loglikelihood))
        weights /= np.sum(weights)
        best_fit = np.average(final_state.particles, weights=weights, axis=0)
        
        # Extract fitted number of pulses
        fitted_npulse = best_fit[-1]
        
        print(f"Fitted number of pulses: {fitted_npulse:.2f}")
        
        # Calculate probability of different numbers of pulses
        npulse_samples = final_state.particles[:, -1]
        for n in range(1, max_peaks + 1):
            prob = np.mean((npulse_samples > n - 0.5) & (npulse_samples <= n + 0.5))
            indicator = " <-- TRUE" if n == self.true_npulse else ""
            print(f"  P(Npulse={n}) = {prob:.3f}{indicator}")
        
        # Check if fitted number is close to true value
        assert abs(fitted_npulse - self.true_npulse) <= 1.0, \
            f"Fitted Npulse={fitted_npulse:.1f} too far from true value {self.true_npulse}"
        
        # Check that the run completed successfully
        assert len(final_state.particles) > 0, "No samples generated"
        assert len(final_state.loglikelihood) == len(final_state.particles), \
            "Mismatch between particles and loglikelihood arrays"
        
        print("✓ Fitted pulse test passed")
        
        return fitted_npulse, best_fit
    
    def test_fixed_pulses(self):
        """Test with fixed number of pulses."""
        max_peaks = 3  # Fix to true number
        fit_pulses = False
        
        print("\nTesting FIXED pulse number estimation...")
        print(f"Fixed number of pulses: {max_peaks}")
        
        # Run nested sampling
        final_state = run_nested_sampling(
            model_name=self.model_name,
            data=self.data_np,
            t=self.t_np,
            prior_bounds=self.prior_bounds,
            max_peaks=max_peaks,
            fit_pulses=fit_pulses,
            **self.ns_params
        )
        
        # Get parameter names
        param_names = get_param_names(self.model_name, max_peaks, fit_pulses)
        
        # Calculate posterior mean
        weights = np.exp(final_state.loglikelihood - np.max(final_state.loglikelihood))
        weights /= np.sum(weights)
        best_fit = np.average(final_state.particles, weights=weights, axis=0)
        
        print(f"Number of parameters: {len(param_names)}")
        print("Parameter estimates:")
        for i, name in enumerate(param_names):
            print(f"  {name}: {best_fit[i]:.3f}")
        
        # Check amplitudes are reasonable
        for i in range(max_peaks):
            fitted_amp = best_fit[i]
            true_amp = self.true_params[i]
            relative_error = abs(fitted_amp - true_amp) / true_amp
            print(f"  Amplitude {i+1} relative error: {relative_error:.2%}")
            assert fitted_amp > 0, f"Amplitude {i+1} should be positive"
            assert fitted_amp < 2.0, f"Amplitude {i+1} unreasonably large"
        
        # Check that the run completed successfully
        assert len(final_state.particles) > 0, "No samples generated"
        assert len(final_state.loglikelihood) == len(final_state.particles), \
            "Mismatch between particles and loglikelihood arrays"
        assert len(param_names) == 3 * max_peaks + 1, \
            f"Expected {3 * max_peaks + 1} parameters, got {len(param_names)}"
        
        print("✓ Fixed pulse test passed")
        
        return best_fit
    
    def test_comparison(self):
        """Compare fitted and fixed results."""
        print("\n" + "="*60)
        print("COMPARING FITTED vs FIXED RESULTS")
        print("="*60)
        
        # Run both tests
        fitted_npulse, fitted_params = self.test_fitted_pulses()
        fixed_params = self.test_fixed_pulses()
        
        print("\nComparison Summary:")
        print(f"  True number of pulses: {self.true_npulse}")
        print(f"  Fitted number estimate: {fitted_npulse:.2f}")
        print(f"  Fixed number used: {self.true_npulse}")
        
        # Compare amplitude estimates (first 3 pulses)
        print("\nAmplitude comparison:")
        for i in range(self.true_npulse):
            true_val = self.true_params[i]
            fitted_val = fitted_params[i]
            fixed_val = fixed_params[i]
            print(f"  A{i+1}: true={true_val:.3f}, fitted={fitted_val:.3f}, fixed={fixed_val:.3f}")
        
        print("\n✓ All tests completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])