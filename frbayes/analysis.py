import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from .utils import downsample, calculate_snr
import scienceplots

# Activate the "science" style
plt.style.use("science")


def plot_inputs(settings):
    # Load data from the HDF5 file
    data = h5py.File(settings["data_file"], "r")
    wfall = data["waterfall"][:]
    data.close()

    # Replace NaN values with the median of the data
    wfall[np.isnan(wfall)] = np.nanmedian(wfall)

    # Extract required parameters
    original_freq_res = float(settings["original_freq_res"])
    original_time_res = float(settings["original_time_res"])
    desired_freq_res = float(settings["desired_freq_res"])
    desired_time_res = float(settings["desired_time_res"])

    # Calculate the downsampling factors
    factor_freq = int(desired_freq_res / original_freq_res)
    factor_time = int(desired_time_res / original_time_res)

    # Downsample the waterfall data
    wfall_downsampled = downsample(wfall, factor_time, factor_freq)

    # Define the min and max values for color scaling
    vmin = np.nanpercentile(wfall_downsampled, 1)
    vmax = np.nanpercentile(wfall_downsampled, 99)

    # Generate frequency and time axis labels for the downsampled data
    num_freq_bins, num_time_bins = wfall_downsampled.shape
    freq_axis = np.linspace(400, 800, num_freq_bins)  # Frequency in MHz
    time_axis = np.arange(num_time_bins) * desired_time_res  # Time in seconds

    # Identify the region where the signal is visible and calculate the pulse profile
    signal_region_indices = (
        np.nanpercentile(wfall_downsampled, 90, axis=1)
        > np.nanmedian(wfall_downsampled)
    ).nonzero()[0]
    pulse_profile = np.nanmean(wfall_downsampled[signal_region_indices, :], axis=0)

    # Calculate Pulse Profile SNR
    pulse_profile_snr, residual_snr = calculate_snr(wfall_downsampled, pulse_profile)

    # Create subplots
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 16), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Plot the waterfall plot
    im = axs[0].imshow(
        wfall_downsampled,
        aspect="auto",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",  # You can change the color map here
        origin="lower",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )

    axs[0].set_ylabel("Frequency (MHz)")
    axs[0].set_title(settings["data_file"])
    axs[0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )

    # Plot the Pulse Profile SNR
    axs[1].plot(time_axis, pulse_profile_snr, label="Pulse Profile SNR", color="k")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Signal / Noise")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/inputs.pdf", bbox_inches="tight")

    plt.show()
