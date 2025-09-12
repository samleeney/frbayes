"""
Test corner plot generation with real nested sampling chains.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frbayes_jax.models import get_model_function, get_param_names
from frbayes_jax.data import simulate_frb_data
from frbayes_jax.sampling import run_nested_sampling

# Check if anesthetic is available
try:
    import anesthetic
    import matplotlib.pyplot as plt
    ANESTHETIC_AVAILABLE = True
except ImportError:
    ANESTHETIC_AVAILABLE = False


class TestPlotting:
    """Test suite for corner plot generation."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test directory and clean up after tests."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="test_plotting_")
        
        # Create sample chain data
        self.create_sample_chains()
        
        yield
        
        # Clean up
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_sample_chains(self):
        """Create real nested sampling chains using a quick 3-pulse exponential run."""
        print("Generating real nested sampling chains...")
        
        # Settings for 3-pulse exponential model
        model_name = "exponential"
        max_peaks = 3
        fit_pulses = False  # Fixed number for simplicity
        
        # True parameters for 3 exponential pulses
        true_params = jnp.array([
            0.8, 0.6, 0.4,    # Amplitudes
            0.4, 0.3, 0.35,   # Tau values
            1.0, 2.0, 3.5,    # Arrival times
            0.05              # Sigma
        ])
        
        # Prior bounds
        prior_bounds = {
            'amplitude': {'min': 0.001, 'max': 1},
            'tau': {'min': 0.1, 'max': 1.0},
            'u': {'min': 0.0, 'max': 5.0},
            'log_sigma': {'min': jnp.log(0.01), 'max': jnp.log(0.5)}
        }
        
        # Simulate data
        model_func = get_model_function(model_name)
        t, data = simulate_frb_data(
            model_func, true_params, max_peaks, fit_pulses,
            t_min=0.0, t_max=5.0, num_points=200,  # Fewer points for speed
            add_noise=True, seed=42
        )
        
        # Convert to numpy
        t_np = np.array(t)
        data_np = np.array(data)
        
        # Run nested sampling with reduced settings for speed
        print("Running quick nested sampling...")
        final_state = run_nested_sampling(
            model_name=model_name,
            data=data_np,
            t=t_np,
            prior_bounds=prior_bounds,
            max_peaks=max_peaks,
            fit_pulses=fit_pulses,
            num_live_points=100,  # Reduced for speed
            num_delete=50,
            num_inner_steps=20,
            log_tolerance=-2.0,
            seed=123
        )
        
        # Store the results
        self.particles = final_state.particles
        self.logL = final_state.loglikelihood
        self.logL_birth = final_state.loglikelihood_birth
        
        # Get parameter names
        self.param_names = get_param_names(model_name, max_peaks, fit_pulses)
        
        # Save chains in anesthetic format
        self.chain_file = os.path.join(self.test_dir, 'test_chains_dead-birth.txt')
        chain_data = np.column_stack([self.particles, self.logL, self.logL_birth])
        np.savetxt(self.chain_file, chain_data)
        
        # Save parameter names
        self.param_file = os.path.join(self.test_dir, 'param_names.txt')
        with open(self.param_file, 'w') as f:
            for name in self.param_names:
                f.write(f"{name}\n")
    
    @pytest.mark.skipif(not ANESTHETIC_AVAILABLE, reason="anesthetic not installed")
    def test_corner_plot(self):
        """Test generating a corner plot with all parameters."""
        # Load the chains
        data = np.loadtxt(self.chain_file)
        particles = data[:, :-2]
        logL = data[:, -2]
        logL_birth = data[:, -1]
        
        # Create NestedSamples object
        ns = anesthetic.NestedSamples(
            data=particles,
            logL=logL,
            logL_birth=logL_birth,
            columns=self.param_names
        )
        
        # Check basic properties
        assert len(ns) > 0, "No samples loaded"
        for param in self.param_names:
            assert param in ns.columns, f"Parameter {param} not in columns"
        
        # Create corner plot with all parameters
        print("Creating corner plot...")
        axes = ns.plot_2d(self.param_names)
        
        # Check that axes were created
        assert axes is not None, "Axes not created"
        
        # Get the current figure
        fig = plt.gcf()
        
        # Save the plot
        plot_file = os.path.join(self.test_dir, 'corner_plot.png')
        fig.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Check that file was created and has content
        assert os.path.exists(plot_file), "Plot file not created"
        assert os.path.getsize(plot_file) > 0, "Plot file is empty"
        
        print(f"✓ Corner plot saved to {plot_file}")
        print("✓ Corner plot test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])