import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from .utils import downsample, calculate_snr
from .data import preprocess_data
import scienceplots

# Activate the "science" style
plt.style.use("science")


def plot_inputs(settings):
    """Plot inputs including the waterfall and pulse profile SNR."""
    # Preprocess data and get pulse profile SNR
    pulse_profile_snr, time_axis = preprocess_data(settings)

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

    # Create subplots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.05)

    # Plot the waterfall plot
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(
        wfall_downsampled,
        aspect="auto",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",  # You can change the color map here
        origin="lower",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )

    ax0.set_ylabel("Frequency (MHz)")
    ax0.set_title(settings["data_file"])
    ax0.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # Plot the Pulse Profile SNR
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(time_axis, pulse_profile_snr, label="Pulse Profile SNR", color="k")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal / Noise")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Adjust layout
    fig.tight_layout()

    # Save the figure
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/inputs.pdf", bbox_inches="tight")
    plt.close()
