import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from .utils import downsample, calculate_snr
from .data import preprocess_data
import scienceplots
from anesthetic import read_chains, make_2d_axes

# Activate the "science" style
plt.style.use("science")


class FRBAnalysis:
    def __init__(self, settings):
        self.settings = settings
        self.pulse_profile_snr, self.time_axis = preprocess_data(settings)
        self.file_root = settings["file_root"]
        self.max_peaks = self.settings["max_peaks"]

        # Define LaTeX-formatted parameter names
        paramnames_all = []
        for i in range(self.max_peaks):
            paramnames_all.append(r"$A_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            paramnames_all.append(r"$\tau_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            paramnames_all.append(r"$u_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            paramnames_all.append(r"$w_{{{}}}$".format(i))

        paramnames_all.append(r"$N_{\text{pulse}}$")
        paramnames_all.append(r"$\sigma$")
        self.paramnames_all = paramnames_all

        # Load the chains
        self.chains = read_chains(
            "chains/" + self.file_root, columns=self.paramnames_all
        )

    def plot_inputs(self):
        """Plot inputs including the waterfall and pulse profile SNR."""
        # Load data from the HDF5 file
        data = h5py.File(self.settings["data_file"], "r")
        wfall = data["waterfall"][:]
        data.close()

        # Replace NaN values with the median of the data
        wfall[np.isnan(wfall)] = np.nanmedian(wfall)

        # Extract required parameters
        original_freq_res = float(self.settings["original_freq_res"])
        original_time_res = float(self.settings["original_time_res"])
        desired_freq_res = float(self.settings["desired_freq_res"])
        desired_time_res = float(self.settings["desired_time_res"])

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
            extent=[self.time_axis[0], self.time_axis[-1], freq_axis[0], freq_axis[-1]],
        )

        ax0.set_ylabel("Frequency (MHz)")
        ax0.set_title(self.settings["data_file"])
        ax0.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        # Plot the Pulse Profile SNR
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax1.plot(
            self.time_axis, self.pulse_profile_snr, label="Pulse Profile SNR", color="k"
        )
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

    def process_chains(self):
        """Process chains with anesthetic and plot posterior distributions."""
        # Enable LaTeX rendering in matplotlib
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        # Select a subset of parameter names to plot
        ptd = 3  # peaks to display
        paramnames_subset = (
            self.paramnames_all[0:ptd]
            + self.paramnames_all[self.max_peaks : self.max_peaks + ptd]
            + self.paramnames_all[2 * self.max_peaks : 2 * self.max_peaks + ptd]
            + self.paramnames_all[3 * self.max_peaks : 3 * self.max_peaks + ptd]
            + self.paramnames_all[4 * self.max_peaks :]
        )

        paramnames_Npulse = [self.paramnames_all[4 * self.max_peaks]]
        paramnames_sigma = [self.paramnames_all[(4 * self.max_peaks) + 1]]
        paramnames_amp = (
            self.paramnames_all[0 : self.max_peaks]
            + paramnames_Npulse
            + paramnames_sigma
        )
        paramnames_tao = (
            self.paramnames_all[self.max_peaks : 2 * self.max_peaks]
            + paramnames_Npulse
            + paramnames_sigma
        )
        paramnames_u = (
            self.paramnames_all[2 * self.max_peaks : 3 * self.max_peaks]
            + paramnames_Npulse
            + paramnames_sigma
        )
        paramnames_w = (
            self.paramnames_all[3 * self.max_peaks : 4 * self.max_peaks]
            + paramnames_Npulse
            + paramnames_sigma
        )

        # Create 2D plot axes ss
        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plot subset...")
        self.chains.plot_2d(ax)
        os.makedirs("results", exist_ok=True)
        fig.savefig(f"results/{self.file_root}_ss_posterior.pdf")
        plt.close()
        print("Done!")
        jk

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

    def functional_posteriors(self):
        from fgivenx import plot_contours
        from frbayes.models import emg

        def emgfgx(t, theta):

            Npulse = theta[4 * self.max_peaks]
            sigma = theta[(4 * self.max_peaks) + 1]
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

        fig, ax = plt.subplots()
        print("Plotting contours...")
        plot_contours(
            emgfgx, self.time_axis, self.chains, ax, weights=self.chains.get_weights()
        )
        ax.set_xlabel("t")
        ax.set_ylabel("SNR")
        plt.savefig(f"results/{self.file_root}_f_posterior.pdf")
        plt.ylim((0, 3))
        plt.show()
        # plt.close()
        print("Done!")
