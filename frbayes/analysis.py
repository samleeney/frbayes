import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from frbayes.utils import downsample, calculate_snr
from frbayes.data import preprocess_data
from frbayes.settings import global_settings  # Import global settings
import scienceplots
from anesthetic import read_chains, make_2d_axes

# Activate the "science" style
plt.style.use("science")


class FRBAnalysis:
    def __init__(self):
        # Access settings globally instead of passing them
        self.pulse_profile_snr, self.time_axis = preprocess_data()
        self.file_root = global_settings.get("file_root")
        self.max_peaks = global_settings.get("max_peaks")

        # Define LaTeX-formatted parameter names
        paramnames_all = []
        for i in range(self.max_peaks):
            paramnames_all.append(r"$A_{{{}}}$".format(i))
            paramnames_all.append(r"$\\tau_{{{}}}$".format(i))
            paramnames_all.append(r"$u_{{{}}}$".format(i))
            paramnames_all.append(r"$w_{{{}}}$".format(i))

        paramnames_all.append(r"$\\sigma$")
        self.paramnames_all = paramnames_all

    def plot_inputs(self):
        """Plot inputs including the waterfall and pulse profile SNR."""
        # Load data from the HDF5 file
        data_file = global_settings.get("data_file")
        data = h5py.File(data_file, "r")
        wfall = data["waterfall"][:]
        data.close()

        # Replace NaN values with the median of the data
        wfall[np.isnan(wfall)] = np.nanmedian(wfall)

        # Extract required parameters directly using global settings
        original_freq_res = float(global_settings.get("original_freq_res"))
        original_time_res = float(global_settings.get("original_time_res"))
        desired_freq_res = float(global_settings.get("desired_freq_res"))
        desired_time_res = float(global_settings.get("desired_time_res"))

        # Calculate the downsampling factors
        factor_freq = int(desired_freq_res / original_freq_res)
        factor_time = int(desired_time_res / original_time_res)

        # Downsample the waterfall data and update other data-related processes
        self.downsampled_wfall = downsample(wfall, factor_time, factor_freq)

    def functional_posteriors(self):
        from fgivenx import plot_contours, plot_lines
        from .models import emg

        def emgfgx(t, theta):
            if global_settings.get("fit_pulses") is True:
                Npulse = theta[(4 * self.max_peaks) + 1]
            else:
                Npulse = self.max_peaks

            sigma = theta[(4 * self.max_peaks)]
            A = theta[0 : self.max_peaks]
            tao = theta[self.max_peaks : 2 * self.max_peaks]
            u = theta[2 * self.max_peaks : 3 * self.max_peaks]
            w = theta[3 * self.max_peaks : 4 * self.max_peaks]
            s = np.zeros((self.max_peaks, len(t)))

            for i in range(self.max_peaks):
                if i < Npulse:
                    s[i] = emg(t, A[i], tao[i], u[i], w[i])
                else:
                    s[i] = 0 * np.ones(len(t))

            return np.sum(s, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

        # Load the chains
        self.chains = read_chains(
            "chains/" + self.file_root, columns=self.paramnames_all
        )

        # Plotting contours and lines
        print("Plotting contours...")
        contour = plot_contours(
            emgfgx,
            self.time_axis,
            self.chains,
            ax1,
            weights=self.chains.get_weights(),
            colors=plt.cm.Blues_r,
        )
        ax1.set_ylabel("SNR")

        print("Plotting lines...")
        lines = plot_lines(
            emgfgx,
            self.time_axis,
            self.chains,
            ax2,
            weights=self.chains.get_weights(),
            color="b",
        )
        ax2.set_xlabel("t")
        ax2.set_ylabel("SNR")

        # Only show the shared y-label once
        fig.text(0.04, 0.5, "SNR", va="center", rotation="vertical")
        fig.tight_layout(rect=[0.05, 0, 1, 1])

        plt.savefig(f"results/{self.file_root}_f_posterior_combined.pdf")
        plt.ylim((0, 3))
        plt.close()
        print("Done!")
