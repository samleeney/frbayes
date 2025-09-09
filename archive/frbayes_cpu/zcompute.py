import os
import anesthetic
import matplotlib.pyplot as plt
import numpy as np

# Define the models and their corresponding directories
model_dirs = {
    "emg": ["chains_emg_1", "chains_emg_2"],
    "exp": ["chains_exponential_1", "chains_exponential_2"],
    "emgT": ["chains_periodic_emg_1", "chains_periodic_emg_2"],
    "expT": ["chains_periodic_exponential_1", "chains_periodic_exponential_2"],
    "double_periodic_exp": [
        "chains_double_periodic_exp_1",
        "chains_double_periodic_exp_2",
    ],
    "periodic_exp_plus_exp": [
        "chains_periodic_exp_plus_exp_1",
        "chains_periodic_exp_plus_exp_2",
    ],
}

# Mapping from model keys to actual names used in filenames
model_names = {
    "emg": "emg",
    "exp": "exponential",
    "emgT": "periodic_emg",
    "expT": "periodic_exponential",
    "double_periodic_exp": "double_periodic_exp",
    "periodic_exp_plus_exp": "periodic_exp_plus_exp",
}

# Models that should be plotted at 2 * n pulses
double_pulse_models = {"double_periodic_exp", "periodic_exp_plus_exp"}

# Initialize data structures
npulse_values = list(range(1, 16))  # n from 1 to 11
evidences_1 = {model: [] for model in model_dirs}
evidences_2 = {model: [] for model in model_dirs}
completeness_1 = {model: [] for model in model_dirs}
completeness_2 = {model: [] for model in model_dirs}

# Specify the number of samples for uncertainty estimation
nsamples = 50

# Loop over npulse values from 1 to 11
for n in npulse_values:
    for model, dirs in model_dirs.items():
        for idx, dir_path in enumerate(dirs):
            # Path to check if the chain is finished
            model_name = model_names[model]
            filename_check = (
                f"fit_pulses=False_{model_name}_npeaks={n}_f_posterior_combined.pdf"
            )
            file_check_path = os.path.join(dir_path, "results", filename_check)

            if os.path.exists(file_check_path):
                # Construct the chain filename
                chain_filename = f"{dir_path}/fit_pulses=False_{model_name}_npeaks={n}"
                # Read the chain
                print(f"Loading chain from {chain_filename}")
                try:
                    chain = anesthetic.read_chains(chain_filename)
                    # Compute evidence and uncertainty
                    logZ = chain.logZ(nsamples)
                    evidence = logZ.mean()
                    uncertainty = logZ.std()
                except Exception as e:
                    print(f"Error loading chain from {chain_filename}: {e}")
                    evidence = np.nan
                    uncertainty = np.nan

                if idx == 0:
                    evidences_1[model].append(evidence)
                    completeness_1[model].append(~np.isnan(evidence))
                else:
                    evidences_2[model].append(evidence)
                    completeness_2[model].append(~np.isnan(evidence))
            else:
                if idx == 0:
                    evidences_1[model].append(np.nan)
                    completeness_1[model].append(False)
                else:
                    evidences_2[model].append(np.nan)
                    completeness_2[model].append(False)

# Convert lists to numpy arrays
npulse_values = np.array(npulse_values)
for model in model_dirs:
    evidences_1[model] = np.array(evidences_1[model])
    evidences_2[model] = np.array(evidences_2[model])
    completeness_1[model] = np.array(completeness_1[model])
    completeness_2[model] = np.array(completeness_2[model])

# Colors and labels for models
colors = {
    "emg": "blue",
    "exp": "green",
    "emgT": "orange",
    "expT": "red",
    "double_periodic_exp": "magenta",
    "periodic_exp_plus_exp": "brown",
}
labels = {
    "emg": "EMG Model",
    "exp": "Exponential Model",
    "emgT": "Periodic EMG Model",
    "expT": "Periodic Exponential Model",
    "double_periodic_exp": "Double Periodic Exponential Model",
    "periodic_exp_plus_exp": "Periodic Exponential + Exponential Model",
}

# Calculate maximum evidence for each model (ignoring NaNs)
max_evidences = {}
for model in model_dirs:
    combined_evidences = np.concatenate([evidences_1[model], evidences_2[model]])
    valid_evidences = combined_evidences[~np.isnan(combined_evidences)]
    max_evidences[model] = valid_evidences.max() if valid_evidences.size > 0 else None

# Plot Bayesian evidence vs npulse for all models
fig, ax1 = plt.subplots(figsize=(14, 9))

for model in model_dirs:
    color = colors[model]
    label = labels[model]

    # Determine x-axis values
    if model in double_pulse_models:
        x_set1 = 2 * npulse_values
        x_set2 = 2 * npulse_values
        chime_x = 18  # 2 * 9
    else:
        x_set1 = npulse_values
        x_set2 = npulse_values
        chime_x = 9  # original CHIME pulse count

    # Plot first directory
    if not np.all(np.isnan(evidences_1[model])):
        # Only label once per model
        ax1.plot(
            x_set1[completeness_1[model]],
            evidences_1[model][completeness_1[model]],
            linestyle="-",
            color=color,
            marker="X",
            markersize=10,
            linewidth=2.5,
            label=label,  # Label only for the first set
        )
        # Plot incomplete chains with 'o'
        ax1.plot(
            x_set1[~completeness_1[model]],
            evidences_1[model][~completeness_1[model]],
            linestyle="None",
            color=color,
            marker="o",
            markersize=8,
            label="_nolegend_",
        )

    # Plot second directory
    if not np.all(np.isnan(evidences_2[model])):
        # Do not add label again
        ax1.plot(
            x_set2[completeness_2[model]],
            evidences_2[model][completeness_2[model]],
            linestyle="--",
            color=color,
            marker="o",
            markersize=10,
            label="_nolegend_",  # No label to avoid duplicate in legend
        )
        # Plot incomplete chains with 'v'
        ax1.plot(
            x_set2[~completeness_2[model]],
            evidences_2[model][~completeness_2[model]],
            linestyle="None",
            color=color,
            marker="v",
            markersize=8,
            label="_nolegend_",
        )

    # Plot maximum evidence horizontal line
    if max_evidences[model] is not None:
        ax1.axhline(
            y=max_evidences[model],
            color=color,
            linestyle=":",
            linewidth=2,
            label="_nolegend_",  # Avoid duplicate legend entries
        )

    # Plot CHIME vertical line
    ax1.axvline(
        x=chime_x,
        color="r",
        linestyle="--",
        linewidth=2,
        label=(
            "CHIME Pulse Count" if model == list(model_dirs.keys())[0] else "_nolegend_"
        ),
    )

# Customize x-axis and y-axis limits
# ax1.set_xlim(left=np.min(npulse_values), right=15)
ax1.set_ylim(bottom=2220)

# Customize x-axis ticks to include both n and 2n where applicable
all_x = []
for n in npulse_values:
    all_x.append(n)
    all_x.append(2 * n)
all_x = sorted(list(set(all_x)))
# Filter x values to be within the x-axis limits
all_x = [x for x in all_x if x <= 15]
ax1.set_xticks(all_x)
ax1.set_xticklabels(all_x, fontsize=12)

# Set labels and title
ax1.set_xlabel("Number of Pulses", fontsize=16)
ax1.set_ylabel(r"$\log Z$", fontsize=16)
ax1.set_title("Bayesian Evidence vs Number of Pulses for Various Models", fontsize=20)

# Handle legend to display each model only once
handles, labels_plot = ax1.get_legend_handles_labels()
# Create a dictionary to remove duplicate labels
by_label = {}
for handle, label in zip(handles, labels_plot):
    if label not in by_label and label != "_nolegend_":
        by_label[label] = handle
ax1.legend(by_label.values(), by_label.keys(), fontsize=12, loc="best")

# Enable grid
ax1.grid(True, linestyle="--", alpha=0.6)

# Adjust layout
plt.tight_layout()

# Save the plot
figure_path = os.path.join("results", "bayesian_evidence_vs_npulse.png")
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
plt.savefig(figure_path)
plt.close()

print(f"Plot saved to {figure_path}")
