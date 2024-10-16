import os
from frbayes import data, analysis, sample
from frbayes.settings import global_settings
from frbayes.sample import FRBModel
import sys


def main():
    """
    Main function to execute the FRB analysis pipeline.

    This function handles the SLURM job ID, loads global settings, sets up the model
    configurations, preprocesses the data, initializes the analysis and model instances,
    runs the PolyChord sampler, and processes and visualizes the results.
    """
    # Handle SLURM job ID or default to 4
    slurm_job_id = (
        4
        if os.environ.get("SLURM_ARRAY_TASK_ID") is None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    )

    # Load global settings
    global_settings.load_settings()
    fit_pulses_ = global_settings.get("fit_pulses")

    # Set the model based on environment variable or default
    model_from_env = os.environ.get("MODEL_FRB")
    if model_from_env is not None:
        global_settings.set("model", model_from_env)
    model_ = global_settings.get("model")
    print("The model is " + model_)

    # Optionally set base_dir from environment variable
    base_dir_from_env = os.environ.get("FRB_BASE_DIR")
    if base_dir_from_env is not None:
        global_settings.set("base_dir", base_dir_from_env)
    base_dir_ = global_settings.get("base_dir")
    print("The base directory is " + base_dir_)

    # Update settings with the maximum number of peaks and file root
    global_settings.set("max_peaks", int(slurm_job_id))
    global_settings.set(
        "file_root",
        f"fit_pulses={fit_pulses_}_{model_}_npeaks={slurm_job_id}",
    )

    # Preprocess data
    data.preprocess_data()

    # Initialize the analysis object and plot inputs
    frb_analysis = analysis.FRBAnalysis()
    # frb_analysis.plot_inputs()

    # Initialize the FRB model instance and run PolyChord sampler
    frb_model = FRBModel()
    frb_model.run_polychord()

    # Process chains and generate functional posteriors
    frb_analysis.process_chains()
    frb_analysis.functional_posteriors()
    frb_analysis.overlay_predictions_on_input()

    # Exit code for slurm
    sys.exit(0)


if __name__ == "__main__":
    main()
