import numpy as np
import matplotlib.pyplot as plt
import os
from frbayes.settings import global_settings
from frbayes.data import preprocess_data
from anesthetic import read_chains, make_2d_axes
from anesthetic.plot import hist_plot_1d
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from frbayes.models import get_model
from fgivenx import plot_contours, plot_lines
import seaborn as sns
import scienceplots

# Activate the "science" style for matplotlib
plt.style.use("science")

# Define colormap and color assignments for each model
MODEL_COLOR_MAPS = {
    "emg": plt.cm.Blues_r,
    "exponential": plt.cm.Greens_r,
    "periodic_exponential": plt.cm.Reds_r,
    "periodic_emg": plt.cm.Oranges_r,
}

MODEL_COLORS = {
    "emg": "blue",
    "exponential": "green",
    "periodic_exponential": "red",
    "periodic_emg": "orange",
}


class FRBAnalysis:
    def __init__(self):
        self.downsampled_wfall, self.pulse_profile_snr, self.time_axis = (
            preprocess_data()
        )
        self.file_root = global_settings.get("file_root")
        self.max_peaks = global_settings.get("max_peaks")
        self.fit_pulses = global_settings.get("fit_pulses")

        model_name = global_settings.get("model")
        print(f"Model name retrieved: {model_name}")

        self.model_name = model_name.strip().lower()

        self.model = get_model(model_name, global_settings)
        self.paramnames_all = self.model.paramnames_all

        self.is_periodic_model = self.model_name in [
            "periodic_exponential",
            "periodic_emg",
        ]

        if self.model_name in MODEL_COLOR_MAPS:
            self.model_color_map = MODEL_COLOR_MAPS[self.model_name]
            self.model_color = MODEL_COLORS[self.model_name]
        else:
            raise KeyError(
                f"Model color map for '{self.model_name}' not found. Available keys: {list(MODEL_COLOR_MAPS.keys())}"
            )

        print(f"Using color map: {self.model_color_map} for model: {model_name}")

    def plot_inputs(self):
        vmin = np.nanpercentile(self.downsampled_wfall, 1)
        vmax = np.nanpercentile(self.downsampled_wfall, 99)

        num_freq_bins, num_time_bins = self.downsampled_wfall.shape
        freq_axis = np.linspace(
            global_settings.get("freq_min"),
            global_settings.get("freq_max"),
            num_freq_bins,
        )  # Frequency in MHz

        fig, axs = plt.subplots(
            2, 1, figsize=(10, 16), gridspec_kw={"height_ratios": [2, 1]}
        )

        im = axs[0].imshow(
            self.downsampled_wfall,
            aspect="auto",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
            cmap=self.model_color_map,  # Use model-specific colormap
            origin="lower",
            extent=[self.time_axis[0], self.time_axis[-1], freq_axis[0], freq_axis[-1]],
        )

        axs[0].set_ylabel("Frequency (MHz)", fontsize=14)
        axs[0].tick_params(axis="both", which="major", labelsize=12)
        axs[0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        axs[1].plot(
            self.time_axis,
            self.pulse_profile_snr,
            label="Pulse Profile SNR",
            color=self.model_color,  # Use model-specific color
        )
        axs[1].set_xlabel("Time (s)", fontsize=14)
        axs[1].set_ylabel("Signal / Noise", fontsize=14)
        axs[1].legend(loc="upper right")
        axs[1].tick_params(axis="both", which="major", labelsize=12)
        axs[1].grid(True)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        fig.savefig("results/inputs.pdf", bbox_inches="tight")
        plt.close()

    def functional_posteriors(self):
        def model_function_wrapper(t, theta):
            if self.fit_pulses:
                Npulse_index = self.model.nDims - 1
                Npulse = int(theta[Npulse_index])
            else:
                Npulse = self.max_peaks

            s = np.zeros((self.max_peaks, len(t)))

            for i in range(self.max_peaks):
                if i < Npulse:
                    s[i] = self.model.model_function(t, theta, i)
                else:
                    s[i] = 0 * np.ones(len(t))

            return np.sum(s, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        print("Plotting contours...")
        plot_contours(
            model_function_wrapper,
            self.time_axis,
            chains,
            ax1,
            weights=chains.get_weights(),
            colors=self.model_color_map,
        )
        ax1.set_ylabel("SNR", fontsize=14)
        ax1.tick_params(axis="both", which="major", labelsize=12)
        print("Done!")

        colors = plt.cm.viridis(np.linspace(0, 1, self.max_peaks))

        if self.is_periodic_model:
            period_param_index = self.model.dim * self.max_peaks + 1
            u0_param_index = self.model.dim * self.max_peaks
            period_chain = chains.iloc[:, period_param_index]
            u0_chain = chains.iloc[:, u0_param_index]
            u_mean = u0_chain.mean()
            period_mean = period_chain.mean()
            period_std = period_chain.std()

            # Plot u_0 + nT lines and spans
            for i in range(self.max_peaks):
                u_n = u0_chain + i * period_chain
                mean = u_n.mean()
                std = u_n.std()
                color = colors[i]

                ax1.axvline(mean, color=color, linestyle="-", linewidth=2)
                ax1.axvspan(
                    mean - std * 0.5,
                    mean + std * 0.5,
                    color=color,
                    alpha=0.3,
                )
                print(f"u_{i}: Mean = {mean}, Std Dev = {std}")
        else:
            for i in range(self.max_peaks):
                mean = chains.loc[
                    :, [self.paramnames_all[2 * self.max_peaks + i]]
                ].mean()
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

        ax1.legend(handles=[mean_handle, std_handle], loc="upper right")

        print("Plotting lines...")
        plot_lines(
            model_function_wrapper,
            self.time_axis,
            chains,
            ax2,
            weights=chains.get_weights(),
            color=self.model_color,
        )
        ax2.set_xlabel("Time (s)", fontsize=14)
        ax2.set_ylabel("SNR", fontsize=14)
        ax2.tick_params(axis="both", which="major", labelsize=12)
        print("Done!")

        fig.tight_layout()
        plt.savefig(f"results/{self.file_root}_f_posterior_combined.pdf")
        plt.close()

    def process_chains(self):
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        ptd = min(self.max_peaks, 7)  # Plot up to 7 peaks

        # Initialize parameter names lists
        paramnames_A = []
        paramnames_tau = []
        paramnames_u = []
        paramnames_w = []
        paramnames_sigma = []
        paramnames_u0 = []
        paramnames_T = []
        paramnames_Npulse = []

        # Assign parameter names based on the model
        if self.is_periodic_model:
            # For periodic models
            dim = self.model.dim
            paramnames_A = self.paramnames_all[: self.max_peaks]
            paramnames_tau = self.paramnames_all[self.max_peaks : 2 * self.max_peaks]

            paramnames_u0 = [self.paramnames_all[2 * self.max_peaks]]
            paramnames_T = [self.paramnames_all[2 * self.max_peaks + 1]]
            paramnames_sigma = [self.paramnames_all[2 * self.max_peaks + 2]]

            if self.fit_pulses:
                paramnames_Npulse = [self.paramnames_all[2 * self.max_peaks + 3]]

            if isinstance(self.model, type(get_model("periodic_emg", global_settings))):
                paramnames_w = self.paramnames_all[
                    2 * self.max_peaks : 3 * self.max_peaks
                ]
                paramnames_u0 = [self.paramnames_all[3 * self.max_peaks]]
                paramnames_T = [self.paramnames_all[3 * self.max_peaks + 1]]
                paramnames_sigma = [self.paramnames_all[3 * self.max_peaks + 2]]

                if self.fit_pulses:
                    paramnames_Npulse = [self.paramnames_all[3 * self.max_peaks + 3]]

            # Construct parameter groups
            paramnames_subset = (
                paramnames_A[:ptd]
                + paramnames_tau[:ptd]
                + paramnames_u0
                + paramnames_T
                + paramnames_sigma
                + paramnames_Npulse
            )
            paramnames_amp = paramnames_A + paramnames_sigma + paramnames_Npulse
            paramnames_tau = paramnames_tau + paramnames_sigma + paramnames_Npulse
            paramnames_u = (
                paramnames_u0 + paramnames_T + paramnames_sigma + paramnames_Npulse
            )
            if paramnames_w:
                paramnames_w = paramnames_w + paramnames_sigma + paramnames_Npulse
        else:
            # For non-periodic models
            paramnames_A = self.paramnames_all[: self.max_peaks]
            paramnames_tau = self.paramnames_all[self.max_peaks : 2 * self.max_peaks]
            paramnames_u = self.paramnames_all[2 * self.max_peaks : 3 * self.max_peaks]
            paramnames_sigma = [self.paramnames_all[self.model.dim * self.max_peaks]]

            if self.fit_pulses:
                paramnames_Npulse = [self.paramnames_all[self.model.nDims - 1]]

            if isinstance(self.model, type(get_model("emg", global_settings))):
                paramnames_w = self.paramnames_all[
                    3 * self.max_peaks : 4 * self.max_peaks
                ]

            # Construct parameter groups
            paramnames_subset = (
                paramnames_A[:ptd]
                + paramnames_tau[:ptd]
                + paramnames_u[:ptd]
                + paramnames_sigma
                + paramnames_Npulse
            )
            paramnames_amp = paramnames_A + paramnames_sigma + paramnames_Npulse
            paramnames_tau = paramnames_tau + paramnames_sigma + paramnames_Npulse
            paramnames_u = paramnames_u + paramnames_sigma + paramnames_Npulse
            if paramnames_w:
                paramnames_w = paramnames_w + paramnames_sigma + paramnames_Npulse

        os.makedirs("results", exist_ok=True)

        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plot subset...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(f"results/{self.file_root}_ss_posterior.pdf")
        plt.close()

        fig, ax = make_2d_axes(paramnames_amp, figsize=(6, 6))
        print("Plot amplitude...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(f"results/{self.file_root}_amp_posterior.pdf")
        plt.close()

        fig, ax = make_2d_axes(paramnames_tau, figsize=(6, 6))
        print("Plot tau...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(f"results/{self.file_root}_tau_posterior.pdf")
        plt.close()

        fig, ax = make_2d_axes(paramnames_u, figsize=(6, 6))
        print("Plot u...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(f"results/{self.file_root}_u_posterior.pdf")
        plt.close()

        if paramnames_w:
            fig, ax = make_2d_axes(paramnames_w, figsize=(6, 6))
            print("Plot w...")
            chains.plot_2d(ax, color=self.model_color)
            ax.tick_params(axis="both", which="major", labelsize=12)
            fig.savefig(f"results/{self.file_root}_w_posterior.pdf")
            plt.close()

        if self.is_periodic_model:
            self.plot_period_distribution(chains)

    def plot_period_distribution(self, chains):
        if self.is_periodic_model:
            if self.model_name == "periodic_exponential":
                period_param_index = 2 * self.max_peaks + 1
            elif self.model_name == "periodic_emg":
                period_param_index = 3 * self.max_peaks + 1
            else:
                raise ValueError("Invalid model name for periodic model.")

            period_param_name = self.paramnames_all[period_param_index]
            period_chain = chains[period_param_name]

            mean_period = period_chain.mean()
            std_period = period_chain.std()

            # Plot using anesthetic
            fig, ax = plt.subplots(figsize=(10, 6))
            hist_plot_1d(
                ax, period_chain, weights=chains.get_weights(), color=self.model_color
            )
            ax.axvline(
                mean_period,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_period:.2f} s",
            )
            ax.axvline(
                mean_period - std_period,
                color="lightgreen",
                linestyle=":",
                linewidth=2,
                label=rf"-1$\sigma$: {mean_period - std_period:.2f} s",
            )
            ax.axvline(
                mean_period + std_period,
                color="lightgreen",
                linestyle=":",
                linewidth=2,
                label=rf"+1$\sigma$: {mean_period + std_period:.2f} s",
            )

            ax.set_xlabel("Period (T) (s)", fontsize=14)
            ax.set_ylabel("Probability Density", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.legend(fontsize=12)
            plt.tight_layout()
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/{self.file_root}_period_distribution.pdf")
            plt.close()

            print("Plotting T...")
        else:
            print("Model is not periodic; period distribution plot not generated.")

    def overlay_predictions_on_input(self):
        chains = read_chains("chains/" + self.file_root, columns=self.paramnames_all)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            self.time_axis,
            self.pulse_profile_snr,
            label="Pulse Profile SNR",
            color="black",
            linewidth=1.5,
            alpha=0.8,
        )

        def model_function_wrapper(t, theta):
            if self.fit_pulses:
                Npulse_index = self.model.nDims - 1
                Npulse = int(theta[Npulse_index])
            else:
                Npulse = self.max_peaks

            s = np.zeros((self.max_peaks, len(t)))

            for i in range(self.max_peaks):
                if i < Npulse:
                    s[i] = self.model.model_function(t, theta, i)
                else:
                    s[i] = 0 * np.ones(len(t))

            return np.sum(s, axis=0)

        print("Overlaying predictions on pulse profile SNR...")
        plot_lines(
            model_function_wrapper,
            self.time_axis,
            chains,
            ax,
            weights=chains.get_weights(),
            color=self.model_color,
            linewidth=1,
            alpha=0.7,
        )

        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Signal / Noise", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{self.file_root}_overlay_predictions_on_snr.pdf")
        plt.close()

        print("Plotting done!")
