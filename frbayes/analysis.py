import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from frbayes.utils import downsample, calculate_snr
from frbayes.data import preprocess_data
from frbayes.settings import global_settings
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

        if global_settings.get("fit_pulses") is True:
            paramnames_all.append(r"$N_{\text{pulse}}$")

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
        from frbayes.models import emg

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

    def process_chains(self):
        """Process chains with anesthetic and plot posterior distributions."""
        # Enable LaTeX rendering in matplotlib
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        self.chains = read_chains(
            "chains/" + self.file_root, columns=self.paramnames_all
        )

        # Select a subset of parameter names to plot
        if self.max_peaks > 3:
            ptd = 3  # peaks to display
        else:
            ptd = np.copy(self.max_peaks)

        paramnames_subset = (
            self.paramnames_all[0:ptd]
            + self.paramnames_all[self.max_peaks : self.max_peaks + ptd]
            + self.paramnames_all[2 * self.max_peaks : 2 * self.max_peaks + ptd]
            + self.paramnames_all[3 * self.max_peaks : 3 * self.max_peaks + ptd]
            + self.paramnames_all[4 * self.max_peaks :]
        )

        paramnames_sigma = [self.paramnames_all[(4 * self.max_peaks)]]
        paramnames_amp = self.paramnames_all[0 : self.max_peaks] + paramnames_sigma
        paramnames_tao = (
            self.paramnames_all[self.max_peaks : 2 * self.max_peaks] + paramnames_sigma
        )
        paramnames_u = (
            self.paramnames_all[2 * self.max_peaks : 3 * self.max_peaks]
            + paramnames_sigma
        )
        paramnames_w = (
            self.paramnames_all[3 * self.max_peaks : 4 * self.max_peaks]
            + paramnames_sigma
        )

        if global_settings.get("file_root") is True:
            paramnames_Npulse = [self.paramnames_all[(4 * self.max_peaks) + 1]]
            paramnames_amp += paramnames_Npulse
            paramnames_tao += paramnames_Npulse
            paramnames_u += paramnames_Npulse
            paramnames_w += paramnames_Npulse

        # Create 2D plot axes ss
        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plot subset...")
        self.chains.plot_2d(ax)
        os.makedirs("results", exist_ok=True)
        fig.savefig(f"results/{self.file_root}_ss_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for amplitude
        fig, ax = make_2d_axes(paramnames_amp, figsize=(6, 6))
        print("Plot amplitude...")
        self.chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_amp_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for tao
        fig, ax = make_2d_axes(paramnames_tao, figsize=(6, 6))
        print("Plot tao...")
        self.chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_tao_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for u
        fig, ax = make_2d_axes(paramnames_u, figsize=(6, 6))
        print("Plot u...")
        self.chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_u_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for w
        fig, ax = make_2d_axes(paramnames_w, figsize=(6, 6))
        print("Plot w...")
        self.chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_w_posterior.pdf")
        plt.close()
        print("Done!")
