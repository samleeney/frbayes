import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from frbayes.utils import downsample, calculate_snr
from frbayes.data import preprocess_data
from frbayes.settings import global_settings
import scienceplots
from anesthetic import read_chains, make_2d_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from frbayes.models import emg, exponential

# Activate the "science" style
plt.style.use("science")


class FRBAnalysis:
    def __init__(self):
        # Access settings globally instead of passing them
        self.downsampled_wfall, self.pulse_profile_snr, self.time_axis = (
            preprocess_data()
        )
        self.file_root = global_settings.get("file_root")
        self.max_peaks = global_settings.get("max_peaks")

        # Define LaTeX-formatted parameter names
        paramnames_all = []
        for i in range(self.max_peaks):
            paramnames_all.append(r"$A_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            paramnames_all.append(r"$\tau_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            paramnames_all.append(r"$u_{{{}}}$".format(i))
        if global_settings.get("model") == "emg":
            for i in range(self.max_peaks):
                paramnames_all.append(r"$w_{{{}}}$".format(i))

        paramnames_all.append(r"$\\sigma$")

        if global_settings.get("fit_pulses") is True:
            paramnames_all.append(r"$N_{\text{pulse}}$")

        self.paramnames_all = paramnames_all
        if global_settings.get("model") == "emg":
            self.model = emg
        elif global_settings.get("model") == "exponential":
            self.model = exponential

    def plot_inputs(self):
        """Plot inputs including the waterfall and pulse profile SNR."""
        # Define the min and max values for color scaling
        vmin = np.nanpercentile(self.downsampled_wfall, 1)
        vmax = np.nanpercentile(self.downsampled_wfall, 99)

        # Generate frequency and time axis labels for the downsampled data
        num_freq_bins, num_time_bins = self.downsampled_wfall.shape
        freq_axis = np.linspace(400, 800, num_freq_bins)  # Frequency in MHz

        # Create subplots
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 16), gridspec_kw={"height_ratios": [2, 1]}
        )

        # Plot the waterfall plot
        im = axs[0].imshow(
            self.downsampled_wfall,
            aspect="auto",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",  # You can change the color map here
            origin="lower",
            extent=[self.time_axis[0], self.time_axis[-1], freq_axis[0], freq_axis[-1]],
        )

        axs[0].set_ylabel("Frequency (MHz)")
        axs[0].set_title(global_settings.get("data_file"))
        axs[0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        # Plot the Pulse Profile SNR
        axs[1].plot(
            self.time_axis, self.pulse_profile_snr, label="Pulse Profile SNR", color="k"
        )
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Signal / Noise")
        axs[1].legend(loc="upper right")
        axs[1].grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        os.makedirs("results", exist_ok=True)
        fig.savefig("results/inputs.pdf", bbox_inches="tight")

        plt.close()

    def functional_posteriors(self):
        from fgivenx import plot_contours, plot_lines
        from frbayes.models import emg

        def emgfgx(t, theta):
            if global_settings.get("model") == "emg":
                if global_settings.get("fit_pulses") is True:
                    Npulse = theta[(4 * self.max_peaks) + 1]
                else:
                    Npulse = self.max_peaks
            elif global_settings.get("model") == "exponential":
                if global_settings.get("fit_pulses") is True:
                    Npulse = theta[(3 * self.max_peaks) + 1]
                else:
                    Npulse = self.max_peaks

            s = np.zeros((self.max_peaks, len(t)))

            for i in range(self.max_peaks):
                if i < Npulse:
                    if global_settings.get("model") == "emg":
                        s[i] = self.model(t, theta, self.max_peaks, i)
                    elif global_settings.get("model") == "exponential":
                        s[i] = self.model(t, theta, self.max_peaks, i)
                else:
                    s[i] = 0 * np.ones(len(t))

            return np.sum(s, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

        # Load the chains
        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        # Plotting contours and lines
        print("Plotting contours...")
        contour = plot_contours(
            emgfgx,
            self.time_axis,
            chains,
            ax1,
            weights=chains.get_weights(),
            colors=plt.cm.Blues_r,
        )
        ax1.set_ylabel("SNR")
        print("Done!")

        colors = plt.cm.viridis(np.linspace(0, 1, self.max_peaks))

        for i in range(0, self.max_peaks):
            mean = chains.loc[:, ["$u_{" + str(i) + "}$"]].mean()
            std = chains.loc[:, ["$u_{" + str(i) + "}$"]].std()
            color = colors[i]

            mean_value = float(mean.iloc[0])
            std_value = float(std.iloc[0])

            ax1.axvline(mean_value, color=color, linestyle="-", linewidth=2)
            ax1.axvspan(
                mean_value - std_value * 0.5,
                mean_value + std_value * 0.5,
                color=color,
                alpha=0.3,
            )

            print(f"Mean is {mean_value}")
            print(f"Standard dev is {std_value}")

        # Create custom legend handles
        mean_handle = Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label=r"$\textsc{mean} \, \mu$",
        )
        std_handle = Patch(
            facecolor="black", edgecolor="black", alpha=0.3, label=r"$1\sigma_\mu$"
        )

        # Add legend to the plot
        ax1.legend(handles=[mean_handle, std_handle], loc="upper right")

        print("Plotting lines...")
        lines = plot_lines(
            emgfgx,
            self.time_axis,
            chains,
            ax2,
            weights=chains.get_weights(),
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

        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        # Select a subset of parameter names to plot
        if self.max_peaks > 3:
            ptd = 3  # peaks to display
        else:
            ptd = np.copy(self.max_peaks)

        paramnames_subset = (
            self.paramnames_all[0:ptd]
            + self.paramnames_all[self.max_peaks : self.max_peaks + ptd]
            + self.paramnames_all[2 * self.max_peaks : 2 * self.max_peaks + ptd]
        )

        if global_settings.get("model") == "emg":
            dim = 4
            paramnames_sigma = [self.paramnames_all[(dim * self.max_peaks)]]
            paramnames_w = (
                self.paramnames_all[3 * self.max_peaks : 4 * self.max_peaks]
                + paramnames_sigma
            )
        elif global_settings.get("model") == "exponential":
            dim = 3
            paramnames_sigma = [self.paramnames_all[(dim * self.max_peaks)]]

        paramnames_subset += self.paramnames_all[dim * self.max_peaks :]
        paramnames_amp = self.paramnames_all[0 : self.max_peaks] + paramnames_sigma
        paramnames_tao = (
            self.paramnames_all[self.max_peaks : 2 * self.max_peaks] + paramnames_sigma
        )
        paramnames_u = (
            self.paramnames_all[2 * self.max_peaks : 3 * self.max_peaks]
            + paramnames_sigma
        )

        if global_settings.get("fit_pulses") is True:
            paramnames_Npulse = [self.paramnames_all[(dim * self.max_peaks) + 1]]
            paramnames_amp += paramnames_Npulse
            paramnames_tao += paramnames_Npulse
            paramnames_u += paramnames_Npulse
            paramnames_w += paramnames_Npulse

        # Create 2D plot axes ss
        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plot subset...")
        chains.plot_2d(ax)
        os.makedirs("results", exist_ok=True)
        fig.savefig(f"results/{self.file_root}_ss_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for amplitude
        fig, ax = make_2d_axes(paramnames_amp, figsize=(6, 6))
        print("Plot amplitude...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_amp_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for tao
        fig, ax = make_2d_axes(paramnames_tao, figsize=(6, 6))
        print("Plot tao...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_tao_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for u
        fig, ax = make_2d_axes(paramnames_u, figsize=(6, 6))
        print("Plot u...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_u_posterior.pdf")
        plt.close()
        print("Done!")

        # Create 2D plot axes for w
        if global_settings.get("model") == "emg":
            fig, ax = make_2d_axes(paramnames_w, figsize=(6, 6))
            print("Plot w...")
            chains.plot_2d(ax)
            fig.savefig(f"results/{self.file_root}_w_posterior.pdf")
            plt.close()
            print("Done!")
