import numpy as np
import yaml


def downsample(data, factor_time, factor_freq):
    return data.reshape(
        data.shape[0] // factor_freq,
        factor_freq,
        data.shape[1] // factor_time,
        factor_time,
    ).mean(axis=(1, 3))


def calculate_snr(wfall_downsampled, pulse_profile):
    # Calculate the standard deviation of the noise
    noise_std = np.nanstd(wfall_downsampled)

    # Calculate Pulse Profile SNR
    pulse_profile_snr = np.atleast_1d(pulse_profile / noise_std)

    # Calculate residual noise
    residual_noise = np.nanstd(wfall_downsampled, axis=0) - np.nanmedian(
        np.nanstd(wfall_downsampled, axis=0)
    )

    # Calculate Residual SNR
    residual_snr = residual_noise / noise_std

    return pulse_profile_snr, residual_snr


def load_settings():
    """Load settings from a YAML file."""
    with open("settings.yaml", "r") as file:
        return yaml.safe_load(file)
