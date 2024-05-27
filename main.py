from frbayes import data, analysis
from frbayes.sample import Sample
from frbayes.utils import load_settings
import importlib
import os
import yaml
import numpy as np

# Reload the modules to ensure changes are reflected
importlib.reload(analysis)
importlib.reload(data)


def main():
    # Load settings
    settings = load_settings()

    slurm_job_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    print("slurm_job_id is " + str(slurm_job_id))
    settings["Npulse"] = int(slurm_job_id)
    settings["file_root"] = "npulse=" + str(settings["Npulse"]) + "_"
    settings["max_peaks"] = np.copy(settings["Npulse"])
    print(f"Running job {slurm_job_id} with Npulse = {settings['Npulse']}")

    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Preprocess data
    data.preprocess_data(settings)

    # Plot inputs
    frb_analysis = analysis.FRBAnalysis(settings)
    frb_analysis.plot_inputs()

    # Run PolyChord to generate the chains
    Sample(settings).run_polychord()  # Assuming a model identifier is needed

    # Process chains with anesthetic
    frb_analysis.process_chains()
    frb_analysis.functional_posteriors()


main()
