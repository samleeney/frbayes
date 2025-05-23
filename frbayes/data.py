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

    preprocessing_mode = global_settings.get("preprocessing", {}).get("mode", "default")

    if preprocessing_mode == "raw":
        # New preprocessing path: no NaN replacement, no downsampling
        final_wfall_output = wfall
        final_time_res_for_axis = original_time_res
        pulse_profile_snr = np.nanmean(final_wfall_output, axis=0)
    elif preprocessing_mode == "paper":
        # Existing "paper preprocessing" logic
        # RFI Mitigation: Replace NaN values with the median of the off-burst region
        # Off-burst region: first 10% of time bins
        off_burst_time_bins = int(wfall.shape[1] * 0.1)
        off_burst_data = wfall[:, :off_burst_time_bins]
        off_burst_median = np.nanmedian(off_burst_data)

        # Calculate the downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)

        # Downsample the waterfall data
        wfall_downsampled = downsample(wfall, factor_time, factor_freq)
        final_wfall_output = wfall_downsampled
        final_time_res_for_axis = desired_time_res

        # S/N Calculation: New method for paper preprocessing
        # Create the 1D mean intensity profile
        pulse_profile_intensity = np.nanmean(final_wfall_output, axis=0)
        pulse_profile_intensity = np.atleast_1d(pulse_profile_intensity)

        # Define off-pulse region for the 1D profile
        num_time_bins_profile = pulse_profile_intensity.shape[0]
        off_pulse_bins_1d = int(num_time_bins_profile * 0.1)

        # Estimate baseline from the 1D profile's off-pulse region
        baseline_1d = np.nanmedian(pulse_profile_intensity[:off_pulse_bins_1d])

        # Subtract baseline
        profile_baseline_subtracted = pulse_profile_intensity - baseline_1d

        # Estimate noise RMS from the baseline-subtracted 1D profile's off-pulse region
        noise_std_1d = np.nanstd(profile_baseline_subtracted[:off_pulse_bins_1d])

        # Add a small epsilon to noise_std_1d if it's zero or NaN
        epsilon = 1e-9
        if np.isnan(noise_std_1d) or noise_std_1d == 0:
            noise_std_1d = epsilon

        # Calculate S/N profile
        pulse_profile_snr = profile_baseline_subtracted / noise_std_1d
    elif preprocessing_mode == "default":
        # Default: Original preprocessing logic
        # Original functionality: Replace NaN values with 0
        wfall[np.isnan(wfall)] = 0

        # Calculate the downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)

        # Downsample the waterfall data
        wfall_downsampled = downsample(wfall, factor_time, factor_freq)
        final_wfall_output = wfall_downsampled
        final_time_res_for_axis = desired_time_res

        # Original functionality: Calculate pulse_profile_snr as mean
        pulse_profile_snr = np.mean(wfall_downsampled, axis=0)
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing_mode}")

    # Ensure pulse_profile_snr is always np.atleast_1d
    pulse_profile_snr = np.atleast_1d(pulse_profile_snr)

    # Generate time axis labels for the processed data
    num_freq_bins, num_time_bins = final_wfall_output.shape
    time_axis = np.arange(num_time_bins) * final_time_res_for_axis  # Time in seconds

    # Save Pulse Profile SNR data to a CSV file
    os.makedirs("results", exist_ok=True)
    np.savetxt("results/pulse_profile_snr.csv", pulse_profile_snr, delimiter=",")
    np.savetxt("results/time_axis.csv", time_axis, delimiter=",")

    return final_wfall_output, pulse_profile_snr, time_axis
