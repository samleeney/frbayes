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
    file_root = "repeating_gaussian"
    # sample.run_polychord(file_root)

    # Process chains with anesthetic
    paramnames = (
        [(f"amplitude_{i}", f"A_{i}") for i in range(3)]
        + [(f"center_{i}", f"c_{i}") for i in range(3)]
        + [(f"width_{i}", f"w_{i}") for i in range(3)]
    )
    print(paramnames)
    jk
    frb_analysis.process_chains(file_root, paramnames)


main()
