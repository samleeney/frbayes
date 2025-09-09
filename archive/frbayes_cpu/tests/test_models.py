import numpy as np
from frbayes.models import get_model
from frbayes.settings import Settings
from frbayes.simulator import simulate_data
import pypolychord
from pypolychord.settings import PolyChordSettings
import os

def run_model_fitting(model_name, data, settings):
    """
    Fit the model to the data using PolyChord and compute Bayesian evidence.
    """
    model = get_model(model_name, settings)
    nDims = model.nDims
    nDerived = 0

    def prior(hypercube):
        return model.prior(hypercube)

    def loglikelihood(theta):
        logL, _ = model.loglikelihood(theta, data)
        return logL, []

    # Configure PolyChord settings for quick testing
    poly_settings = PolyChordSettings(nDims, nDerived)
    poly_settings.file_root = f"chains_{model_name}"
    poly_settings.base_dir = "./chains"
    poly_settings.nlive = 25  # Very small number for quick testing
    poly_settings.num_repeats = 5  # Minimum repeats for quick testing
    poly_settings.do_clustering = False  # Disable clustering for speed
    poly_settings.read_resume = False
    poly_settings.write_resume = False
    poly_settings.num_prior = 10
    poly_settings.use_mpi = False

    # Ensure the output directory exists
    if not os.path.exists(poly_settings.base_dir):
        os.makedirs(poly_settings.base_dir)

    # Run PolyChord
    output = pypolychord.run_polychord(
        loglikelihood, nDims, nDerived, poly_settings, prior
    )

    return output.logZ

def main():
    """
    Main function to test model fitting and evidence computation.
    """
    # For basic testing, just test EMG vs Exponential
    model_names = [
        'emg',
        'exponential',
    ]

    # Simulate data using EMG model
    true_model_name = 'emg'
    settings = Settings()
    settings.set("max_peaks", 2)
    settings.set("fit_pulses", False)

    # Define true parameters for EMG model
    theta_true = np.array([
        0.8, 0.5,             # A1, A2
        1.0, 0.5,             # tau1, tau2
        1.0, 3.0,             # u1, u2
        0.1, 0.1,             # w1, w2
        0.05                  # sigma
    ])

    # Simulate data with fewer points for quick testing
    print("Simulating data...")
    data = simulate_data(true_model_name, theta_true, num_points=50)

    # Fit each model to the simulated data
    logZ_dict = {}
    for model_name in model_names:
        print(f"\nFitting model: {model_name}")
        logZ = run_model_fitting(model_name, data, settings)
        logZ_dict[model_name] = logZ
        print(f"Model: {model_name}, LogZ: {logZ}")

    # Print results
    print("\nResults:")
    for model_name, logZ in logZ_dict.items():
        print(f"{model_name}: logZ = {logZ}")

    # Determine the best model
    best_model = max(logZ_dict, key=logZ_dict.get)
    print(f"\nBest model based on Bayesian evidence: {best_model}")

    # Verify that the true model is selected
    assert best_model == true_model_name, f"Model selection failed. Expected {true_model_name}, got {best_model}."
    print("\nTest passed! The true model (EMG) was correctly identified.")

if __name__ == "__main__":
    main() 