from frbayes import data, analysis, sample, models
from frbayes.settings import global_settings
import importlib
import os
import yaml
import numpy as np


def main():
    # Ensure the global settings are loaded, handled by global_settings instance in settings.py
    global_settings.load_settings()

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Preprocess data
    data.preprocess_data()  # Assuming global settings are now accessed within this function

    # Plot inputs
    frb_analysis = (
        analysis.FRBAnalysis()
    )  # Assuming global settings are now accessed within this class
    frb_analysis.plot_inputs()

    # Run PolyChord to generate the chains
    file_root = "simple_gaussian"
    sample.run_polychord(file_root)  # Assuming a model identifier is needed

    # Process chains with anesthetic
    frb_analysis.process_chains()
    frb_analysis.functional_posteriors()


main()
