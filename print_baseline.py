#!/usr/bin/env python3

import numpy as np
import h5py
from frbayes.settings import global_settings

def main():
    # 1. Load global settings from frbayes/settings.py (already done by importing global_settings)

    # 2. Read the raw waterfall data from the HDF5 file specified in the settings (data_file).
    data_file_path = global_settings.get("data_file")
    if not data_file_path:
        print("Error: 'data_file' not found in settings.yaml")
        return

    try:
        with h5py.File(data_file_path, "r") as data:
            wfall = data["waterfall"][:]
    except FileNotFoundError:
        print(f"Error: HDF5 data file not found at {data_file_path}")
        return
    except KeyError:
        print(f"Error: 'waterfall' dataset not found in {data_file_path}")
        return

    # 3. Computes the off-pulse median exactly as in the “paper” preprocessing (first 10% of time bins).
    off_burst_time_bins = int(wfall.shape[1] * 0.1)
    off_burst_data = wfall[:, :off_burst_time_bins]
    off_pulse_median = np.nanmedian(off_burst_data)

    # Downsample to a 1D profile
    pulse_profile_intensity = np.nanmean(wfall, axis=0)

    # Subtract off_burst_median to get profile_baseline_subtracted
    profile_baseline_subtracted = pulse_profile_intensity - off_pulse_median

    # Compute the noise standard deviation on the off-pulse region
    noise_std = np.nanstd(profile_baseline_subtracted[:off_burst_time_bins])

    # 4. Prints the off-pulse median.
    print(f"Off-pulse median: {off_pulse_median}")

    # Print the off-pulse noise standard deviation
    print(f"Off-pulse noise std: {noise_std}")

    # 5. Reads the baseline_offset prior range from settings (prior_ranges -> emg_with_baseline -> baseline_offset).
    # 6. Prints the baseline offset range (min and max).
    baseline_offset_prior = global_settings.get_prior_range("emg_with_baseline", "baseline_offset")

    if baseline_offset_prior:
        min_val = baseline_offset_prior.get("min")
        max_val = baseline_offset_prior.get("max")
        print(f"Baseline offset prior: min={min_val}, max={max_val}")
    else:
        print("Baseline offset prior: Not found in settings.")

if __name__ == "__main__":
    main()