import os
from frbayes import data, analysis, sample
from frbayes.settings import global_settings
from frbayes.sample import FRBModel  # Import FRBModel class from frb_model.py


def main():
    slurm_job_id = (
        5
        if os.environ.get("SLURM_ARRAY_TASK_ID") is None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    )
    global_settings.load_settings()
    fit_pulses_ = global_settings.get("fit_pulses")
    global_settings.set("model", str(os.environ.get("MODEL_FRB")))
    model_ = global_settings.get("model")
    print("The model is " + model_)
    global_settings.set("max_peaks", int(slurm_job_id))
    global_settings.set(
        "file_root",
        f"fit_pulses={fit_pulses_}_{model_}_npeaks={slurm_job_id}",
    )

    # Preprocess data
    data.preprocess_data()

    # # Plot inputs
    frb_analysis = analysis.FRBAnalysis()
    frb_analysis.plot_inputs()

    # Initialize the FRB model and run PolyChord
    frb_model = FRBModel()
    frb_model.run_polychord()

    # Process chains with anesthetic using the frb_analysis instance
    frb_analysis.process_chains()
    frb_analysis.functional_posteriors()


if __name__ == "__main__":
    main()
