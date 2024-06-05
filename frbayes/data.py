import numpy as np
import h5py
import os
from frbayes.utils import downsample, calculate_snr
from frbayes.settings import global_settings


def preprocess_data():
    """Preprocess data and save pulse profile SNR to CSV."""
    # Load data from the HDF5 file
    data = h5py.File(global_settings.get("data_file"), "r")

    original_freq_res = float(global_settings.get("original_freq_res"))
    original_time_res = float(global_settings.get("original_time_res"))
    desired_freq_res = float(global_settings.get("desired_freq_res"))
    desired_time_res = float(global_settings.get("desired_time_res"))

    wfall = data["waterfall"][:]
    data.close()

    # Replace NaN values with the median of the data
    wfall[np.isnan(wfall)] = np.nanmedian(wfall)

    # Extract required parameters

    # Calculate the downsampling factors
    factor_freq = int(desired_freq_res / original_freq_res)
    factor_time = int(desired_time_res / original_time_res)

    # Downsample the waterfall data
    wfall_downsampled = downsample(wfall, factor_time, factor_freq)

    # Generate time axis labels for the downsampled data
    num_freq_bins, num_time_bins = wfall_downsampled.shape
    time_axis = np.arange(num_time_bins) * desired_time_res  # Time in seconds

    # Identify the region where the signal is visible and calculate the pulse profile
    signal_region_indices = (
        np.nanpercentile(wfall_downsampled, 90, axis=1)
        > np.nanmedian(wfall_downsampled)
    ).nonzero()[0]
    pulse_profile = np.nanmean(wfall_downsampled[signal_region_indices, :], axis=0)

    # Calculate Pulse Profile SNR
    pulse_profile_snr, _ = calculate_snr(wfall_downsampled, pulse_profile)

    # Save Pulse Profile SNR data to a CSV file
    os.makedirs("results", exist_ok=True)
    np.savetxt("results/pulse_profile_snr.csv", pulse_profile_snr, delimiter=",")
    np.savetxt("results/time_axis.csv", time_axis, delimiter=",")

    return pulse_profile_snr, time_axis
