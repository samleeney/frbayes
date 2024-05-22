import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from .utils import downsample, calculate_snr
from .data import preprocess_data
import scienceplots

# Activate the "science" style
plt.style.use("science")


class FRBAnalysis:
    def __init__(self, settings):
        self.settings = settings
        self.pulse_profile_snr, self.time_axis = preprocess_data(settings)

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

    def process_chains(self, file_root):
        """Process chains with anesthetic and plot posterior distributions."""
        from anesthetic import read_chains, make_2d_axes
        import matplotlib.pyplot as plt

        # Enable LaTeX rendering in matplotlib
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        max_peaks = self.settings["max_peaks"]
        nDims = max_peaks * 4 + 2

        # Define LaTeX-formatted parameter names
        paramnames_all = []
        for i in range(max_peaks):
            paramnames_all.append(f"$A_{{{i}}}$")
        for i in range(max_peaks):
            paramnames_all.append(f"$\\lambda_{{{i}}}$")
        for i in range(max_peaks):
            paramnames_all.append(f"$u_{{{i}}}$")
        for i in range(max_peaks):
            paramnames_all.append(f"$w_{{{i}}}$")
        # for i in range(10):
        #     paramnames_all.append(f"$\\sigma_{{\\text{{pulse}}, {i}}}$")
        paramnames_all.append("$N_{\\text{pulse}}$")
        paramnames_all.append("$\\sigma$")

        # Select a subset of parameter names to plot
        ptd = 3  # peaks to display
        paramnames_subset = (
            paramnames_all[0:ptd]
            + paramnames_all[max_peaks : max_peaks + ptd]
            + paramnames_all[2 * max_peaks : max_peaks + ptd]
            + paramnames_all[3 * max_peaks : max_peaks + ptd]
            + paramnames_all[4 * max_peaks :]
        )
        # paramnames_subset = paramnames_all

        # Load the chains
        chains = read_chains("chains/" + file_root, columns=paramnames_all)

        # Create 2D plot axes
        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plotting...")

        # Plot the chains
        chains.plot_2d(ax)
        print("Done!")

        # Save the plot
        os.makedirs("results", exist_ok=True)
        fig.savefig(f"results/{file_root}_posterior.pdf")
        plt.close()
