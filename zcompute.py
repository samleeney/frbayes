import os
import anesthetic
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the files
directory = "chains"

# Initialize lists to store npulse values, Bayesian evidences, and their uncertainties
npulse_values = []
bayesian_evidences_emg = []
bayesian_evidences_exp = []
bayesian_evidences_emgT = []
bayesian_evidences_expT = []
uncertainties_emg = []
uncertainties_exp = []
uncertainties_emgT = []
uncertainties_expT = []

# Specify the number of samples for uncertainty estimation
nsamples = 50

# Loop over n in the range from 1 to 15
for n in range(1, 12):
    # Construct the filenames
    filename_emg = f"chains/fit_pulses=False_emg_npeaks={n}"
    filename_exp = f"chains/fit_pulses=False_exponential_npeaks={n}"
    filename_emgT = f"chains/fit_pulses=False_periodic_emg_npeaks={n}"
    filename_expT = f"chains/fit_pulses=False_periodic_exponential_npeaks={n}"

    # Process the 'emg' file
    print(f"Loading chain from {filename_emg}")
    chain_emg = anesthetic.read_chains(filename_emg)
    evidence_emg = chain_emg.logZ(nsamples).mean()
    uncertainty_emg = chain_emg.logZ(nsamples).std()
    bayesian_evidences_emg.append(evidence_emg)
    uncertainties_emg.append(uncertainty_emg)

    # Process the 'exp' file
    print(f"Loading chain from {filename_exp}")
    chain_exp = anesthetic.read_chains(filename_exp)
    evidence_exp = chain_exp.logZ(nsamples).mean()
    uncertainty_exp = chain_exp.logZ(nsamples).std()
    bayesian_evidences_exp.append(evidence_exp)
    uncertainties_exp.append(uncertainty_exp)

    # Process the 'emgT' file
    print(f"Loading chain from {filename_emgT}")
    chain_emgT = anesthetic.read_chains(filename_emgT)
    evidence_emgT = chain_emgT.logZ(nsamples).mean()
    uncertainty_emgT = chain_emgT.logZ(nsamples).std()
    bayesian_evidences_emgT.append(evidence_emgT)
    uncertainties_emgT.append(uncertainty_emgT)

    # Process the 'expT' file
    print(f"Loading chain from {filename_expT}")
    chain_expT = anesthetic.read_chains(filename_expT)
    evidence_expT = chain_expT.logZ(nsamples).mean()
    uncertainty_expT = chain_expT.logZ(nsamples).std()
    bayesian_evidences_expT.append(evidence_expT)
    uncertainties_expT.append(uncertainty_expT)

    npulse_values.append(n)

# Convert lists to numpy arrays for easier processing
npulse_values = np.array(npulse_values)
bayesian_evidences_emg = np.array(bayesian_evidences_emg)
bayesian_evidences_exp = np.array(bayesian_evidences_exp)
bayesian_evidences_emgT = np.array(bayesian_evidences_emgT)
bayesian_evidences_expT = np.array(bayesian_evidences_expT)
uncertainties_emg = np.array(uncertainties_emg)
uncertainties_exp = np.array(uncertainties_exp)
uncertainties_emgT = np.array(uncertainties_emgT)
uncertainties_expT = np.array(uncertainties_expT)

# Find the index of the maximum evidence value
max_index_emg = np.argmax(bayesian_evidences_emg)
max_index_exp = np.argmax(bayesian_evidences_exp)
max_index_emgT = np.argmax(bayesian_evidences_emgT)
max_index_expT = np.argmax(bayesian_evidences_expT)

# Plot Bayesian evidence vs npulse for all sets of filenames
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(
    npulse_values,
    bayesian_evidences_emg,
    linestyle="-",
    color="blue",  # EMG model color
    marker="o",
    markersize=13,  # larger markers
    linewidth=3.5,  # thicker lines
    label="EMG model",
)
ax1.plot(
    npulse_values,
    bayesian_evidences_exp,
    linestyle="-",
    color="green",  # Exponential model color
    marker="o",
    markersize=13,  # larger markers
    linewidth=3.5,  # thicker lines
    label="Exponential Model",
)
ax1.plot(
    npulse_values,
    bayesian_evidences_emgT,
    linestyle="-",
    color="orange",  # Periodic EMG model color
    marker="o",
    markersize=13,  # larger markers
    linewidth=3.5,  # thicker lines
    label="Periodic EMG model",
)
ax1.plot(
    npulse_values,
    bayesian_evidences_expT,
    linestyle="-",
    color="red",  # Periodic Exponential model color
    marker="o",
    markersize=13,  # larger markers
    linewidth=3.5,  # thicker lines
    label="Periodic Exponential Model",
)

# Highlight the maximum evidence points
ax1.axvline(
    x=npulse_values[max_index_emg],
    color="k",
    linestyle="--",
    linewidth=2,
    label=r"Max $\log Z$",
)


ax1.axvline(
    x=9,
    color="r",
    linestyle="--",
    linewidth=2,
    label=r"CHIME Pulse Count",
)

# Add horizontal lines with logZ values
ax1.axhline(
    y=bayesian_evidences_emg[max_index_emg],
    color="blue",
    linestyle="--",
    linewidth=2,
)
ax1.axhline(
    y=bayesian_evidences_exp[max_index_exp],
    color="green",
    linestyle="--",
    linewidth=2,
)
ax1.axhline(
    y=bayesian_evidences_emgT[max_index_emgT],
    color="orange",
    linestyle="--",
    linewidth=2,
)
ax1.axhline(
    y=bayesian_evidences_expT[max_index_expT],
    color="red",
    linestyle="--",
    linewidth=2,
)

# Create a secondary y-axis for labels
ax2 = ax1.twinx()
ax2.set_yticks(
    [
        bayesian_evidences_emg[max_index_emg],
        bayesian_evidences_exp[max_index_exp],
        bayesian_evidences_emgT[max_index_emgT],
        bayesian_evidences_expT[max_index_expT],
    ]
)
ax2.set_yticklabels(
    [
        f"{bayesian_evidences_emg[max_index_emg]:.0f}",
        f"{bayesian_evidences_exp[max_index_exp]:.0f}",
        f"{bayesian_evidences_emgT[max_index_emgT]:.0f}",
        f"{bayesian_evidences_expT[max_index_expT]:.0f}",
    ],
    fontsize=18,
)
ax2.set_ylim(ax1.get_ylim())

ax1.set_xlabel("Number of Pulses", fontsize=22)
ax1.set_ylabel(r"$\log Z$", fontsize=22)
ax1.set_xticks(npulse_values)  # Set x-tick locations
ax1.set_xticklabels(npulse_values, fontsize=18)  # Set x-tick labels
ax1.yaxis.set_tick_params(labelsize=18)  # Set y-tick label size on the primary y-axis
ax1.legend(fontsize=18)
ax1.grid(True)
plt.tight_layout()

# Save the plot
figure_path = os.path.join("results", "bayesian_evidence_vs_npulse.png")
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
plt.savefig(figure_path)
plt.close()
