import yaml
from frbayes import data, analysis, sample
import importlib
import os

# Reload the modules to ensure changes are reflected
importlib.reload(analysis)
importlib.reload(data)
importlib.reload(sample)


def load_settings(yaml_file):
    """Load settings from a YAML file."""
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def main():
    # Load settings
    settings = load_settings("settings.yaml")

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Preprocess data
    data.preprocess_data(settings)

    # Plot inputs
    frb_analysis = analysis.FRBAnalysis(settings)
    frb_analysis.plot_inputs()

    # Run PolyChord to generate the chains
    file_root = "simple_gaussian"
    # sample.run_polychord(file_root)  # Assuming a model identifier is needed

    # Process chains with anesthetic
    frb_analysis.process_chains(file_root)


main()
