import numpy as np
import matplotlib.pyplot as plt
import os
from frbayes.settings import global_settings
from frbayes.data import preprocess_data
from anesthetic import read_chains, make_2d_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from frbayes.models import get_model
import seaborn as sns
import scienceplots

# Activate the "science" style for matplotlib
plt.style.use("science")


class FRBAnalysis:
    """
    Class for data analysis and result visualization.

    This class handles the visualization of the inputs, processing of the PolyChord
    chains, plotting of posterior distributions, and plotting of functional posteriors.

    Attributes:
        downsampled_wfall (array): Downsampled waterfall data.
        pulse_profile_snr (array): Pulse profile SNR.
        time_axis (array): Time axis corresponding to the data.
        file_root (str): Root name for output files.
        max_peaks (int): Maximum number of peaks to consider in the model.
        fit_pulses (bool): Whether to fit the number of pulses (peaks).
        model (BaseModel): The model instance used for analysis.
        paramnames_all (list): List of parameter names for plotting.
    """

    def __init__(self):
        # Preprocess data and retrieve the necessary attributes
        self.downsampled_wfall, self.pulse_profile_snr, self.time_axis = (
            preprocess_data()
        )
        self.file_root = global_settings.get("file_root")
        self.max_peaks = global_settings.get("max_peaks")
        self.fit_pulses = global_settings.get("fit_pulses")

        # Initialize the model instance
        model_name = global_settings.get("model")
        self.model = get_model(model_name, global_settings)
        self.paramnames_all = self.model.paramnames_all

    def plot_inputs(self):
        """
        Plot the input data including the waterfall plot and pulse profile SNR.

        This function generates and saves a figure with the waterfall plot on the top
        and the pulse profile SNR on the bottom.
        """
        # Define the min and max values for color scaling
        vmin = np.nanpercentile(self.downsampled_wfall, 1)
        vmax = np.nanpercentile(self.downsampled_wfall, 99)

        # Generate frequency axis labels for the downsampled data
        num_freq_bins, num_time_bins = self.downsampled_wfall.shape
        freq_axis = np.linspace(
            global_settings.get("freq_min"),
            global_settings.get("freq_max"),
            num_freq_bins,
        )  # Frequency in MHz

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
            cmap="viridis",
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
        """
        Plot the functional posteriors using samples from the model.

        This function generates and saves a figure showing the posterior predictive
        contours and lines, as well as the mean and standard deviation of the model
        parameters overlaid on the data.
        """
        from fgivenx import plot_contours, plot_lines

        def model_function_wrapper(t, theta):
            """
            Wrapper function to compute the sum of model functions for given parameters.

            Args:
                t (array): Time axis.
                theta (array): Model parameters.

            Returns:
                array: Summed model function evaluated at time t.
            """
            # Determine the number of pulses
            if self.fit_pulses:
                Npulse_index = self.model.nDims - 1
                Npulse = int(theta[Npulse_index])
            else:
                Npulse = self.max_peaks

            # Initialize the summed signal
            s = np.zeros((self.max_peaks, len(t)))

            # Sum over the peaks
            for i in range(self.max_peaks):
                if i < Npulse:
                    s[i] = self.model.model_function(t, theta, i)
                else:
                    s[i] = 0 * np.ones(len(t))

            return np.sum(s, axis=0)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

        # Load the chains
        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        # Plot posterior predictive contours
        print("Plotting contours...")
        contour = plot_contours(
            model_function_wrapper,
            self.time_axis,
            chains,
            ax1,
            weights=chains.get_weights(),
            colors=plt.cm.Blues_r,
        )
        ax1.set_ylabel("SNR")
        print("Done!")

        # Colors for mean and std lines
        colors = plt.cm.viridis(np.linspace(0, 1, self.max_peaks))

        # Plot mean and standard deviation for each peak
        for i in range(self.max_peaks):
            mean = chains.loc[:, [self.paramnames_all[2 * self.max_peaks + i]]].mean()
            std = chains.loc[:, [self.paramnames_all[2 * self.max_peaks + i]]].std()
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

            print(f"Peak {i}: Mean = {mean_value}, Std Dev = {std_value}")

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

        # Plot posterior predictive lines
        print("Plotting lines...")
        lines = plot_lines(
            model_function_wrapper,
            self.time_axis,
            chains,
            ax2,
            weights=chains.get_weights(),
            color=self.model.color,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("SNR")
        print("Done!")

        # Adjust layout and save the figure
        fig.tight_layout()
        plt.savefig(f"results/{self.file_root}_f_posterior_combined.pdf")
        plt.close()

    def process_chains(self):
        """
        Process chains with anesthetic and plot posterior distributions.

        This function generates and saves corner plots for subsets of parameters,
        including amplitudes, time constants, locations, and any additional parameters.
        """
        # Enable LaTeX rendering in matplotlib
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        # Load the chains
        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        # Determine number of peaks to display
        ptd = min(self.max_peaks, 7)  # Peaks to display

        # Build parameter subsets for plotting
        paramnames_subset = (
            self.paramnames_all[0:ptd]
            + self.paramnames_all[self.max_peaks : self.max_peaks + ptd]
            + self.paramnames_all[2 * self.max_peaks : 2 * self.max_peaks + ptd]
        )
        paramnames_sigma = [self.paramnames_all[self.model.dim * self.max_peaks]]
        paramnames_amp = self.paramnames_all[0 : self.max_peaks] + paramnames_sigma
        paramnames_tau = (
            self.paramnames_all[self.max_peaks : 2 * self.max_peaks] + paramnames_sigma
        )
        paramnames_u = (
            self.paramnames_all[2 * self.max_peaks : 3 * self.max_peaks]
            + paramnames_sigma
        )

        # Include additional parameters based on the model
        if isinstance(self.model, type(get_model("emg", global_settings))):
            paramnames_w = (
                self.paramnames_all[3 * self.max_peaks : 4 * self.max_peaks]
                + paramnames_sigma
            )
            paramnames_subset += paramnames_w[0:ptd]

        if self.fit_pulses:
            paramnames_Npulse = [self.paramnames_all[self.model.nDims - 1]]
            paramnames_amp += paramnames_Npulse
            paramnames_tau += paramnames_Npulse
            paramnames_u += paramnames_Npulse
            if isinstance(self.model, type(get_model("emg", global_settings))):
                paramnames_w += paramnames_Npulse

        # Plot subsets of parameters
        os.makedirs("results", exist_ok=True)

        # Subset plot
        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plot subset...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_ss_posterior.pdf")
        plt.close()
        print("Done!")

        # Amplitude plot
        fig, ax = make_2d_axes(paramnames_amp, figsize=(6, 6))
        print("Plot amplitude...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_amp_posterior.pdf")
        plt.close()
        print("Done!")

        # Time constant plot
        fig, ax = make_2d_axes(paramnames_tau, figsize=(6, 6))
        print("Plot tau...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_tau_posterior.pdf")
        plt.close()
        print("Done!")

        # Location (u) plot
        fig, ax = make_2d_axes(paramnames_u, figsize=(6, 6))
        print("Plot u...")
        chains.plot_2d(ax)
        fig.savefig(f"results/{self.file_root}_u_posterior.pdf")
        plt.close()
        print("Done!")

        # Width plot (for EMG model)
        if isinstance(self.model, type(get_model("emg", global_settings))):
            fig, ax = make_2d_axes(paramnames_w, figsize=(6, 6))
            print("Plot w...")
            chains.plot_2d(ax)
            fig.savefig(f"results/{self.file_root}_w_posterior.pdf")
            plt.close()
            print("Done!")

    def plot_peak_period_distribution(self):
        """
        Plot the distribution of periods between predicted peak locations (u values).

        This function generates and saves a histogram of the time differences between
        successive peaks, along with the mean and standard deviation.
        """
        # Load the chains
        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        # Extract and sort predicted u values from chains
        u_columns = [
            self.paramnames_all[2 * self.max_peaks + i] for i in range(self.max_peaks)
        ]
        u_values = chains.loc[:, u_columns]

        # Sort the u_values for each sample to ensure increasing order
        u_values_sorted = np.sort(u_values.values, axis=1)

        # Calculate the periods (differences between successive u values)
        peak_periods = np.diff(
            u_values_sorted, axis=1
        )  # Shape: (n_samples, max_peaks - 1)

        # Flatten the array for plotting
        peak_periods_flat = peak_periods.flatten()

        # Remove NaN and zero values
        peak_periods_flat = peak_periods_flat[~np.isnan(peak_periods_flat)]
        peak_periods_flat = peak_periods_flat[peak_periods_flat > 0]

        # Calculate statistics
        mean_period = np.mean(peak_periods_flat)
        std_period = np.std(peak_periods_flat)

        # Plot the probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(peak_periods_flat, bins=50, kde=True, color="mediumseagreen")

        # Add vertical lines for mean and 1-sigma deviations
        plt.axvline(
            mean_period,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_period:.2f} s",
        )
        plt.axvline(
            mean_period - std_period,
            color="lightgreen",
            linestyle=":",
            linewidth=2,
            label=f"-1σ: {mean_period - std_period:.2f} s",
        )
        plt.axvline(
            mean_period + std_period,
            color="lightgreen",
            linestyle=":",
            linewidth=2,
            label=f"+1σ: {mean_period + std_period:.2f} s",
        )

        plt.xlabel("Period between Peaks (s)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Periods Between Predicted Peaks")
        plt.legend()
        plt.grid(True)

        # Save the plot
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{self.file_root}_peak_period_distribution.pdf")
        plt.close()

        print("Period distribution plot saved!")
