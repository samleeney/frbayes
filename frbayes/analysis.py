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
    "double_periodic_exp": plt.cm.Purples_r,
    "periodic_exp_plus_exp": plt.cm.YlOrBr_r,
}

MODEL_COLORS = {
    "emg": "blue",
    "exponential": "green",
    "periodic_exponential": "red",
    "periodic_emg": "orange",
    "double_periodic_exp": "magenta",
    "periodic_exp_plus_exp": "brown",
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
            "double_periodic_exp",
            "periodic_exp_plus_exp",
        ]

        if self.model_name in MODEL_COLOR_MAPS:
            self.model_color_map = MODEL_COLOR_MAPS[self.model_name]
            self.model_color = MODEL_COLORS[self.model_name]
        else:
            raise KeyError(
                f"Model color map for '{self.model_name}' not found. Available keys: {list(MODEL_COLOR_MAPS.keys())}"
            )

        print(f"Using color map: {self.model_color_map} for model: {model_name}")

        # Set results directory based on base_dir and file_root
        base_dir = global_settings.get("base_dir")
        self.results_dir = os.path.join(base_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

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
        fig.savefig(os.path.join(self.results_dir, "inputs.png"), bbox_inches="tight")
        plt.close()

    def functional_posteriors(self):
        def model_function_wrapper(t, theta):
            # Check if the model is one of the combined models
            if self.model_name in ["double_periodic_exp", "periodic_exp_plus_exp"]:
                pp_, f1, f2 = self.model.model_function(t, theta)
                return pp_
            else:
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

        chains = read_chains(
            os.path.join(global_settings.get("base_dir"), self.file_root),
            columns=self.paramnames_all,
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 15),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1]},
        )

        print("Plotting combined contours...")
        # Combined model posterior
        plot_contours(
            model_function_wrapper,
            self.time_axis,
            chains,
            axes[0],
            weights=chains.get_weights(),
            colors=self.model_color_map,
        )
        axes[0].set_ylabel("SNR", fontsize=14)
        axes[0].tick_params(axis="both", which="major", labelsize=12)
        axes[0].set_title("Combined Model", fontsize=16)
        print("Done!")

        if self.model_name in ["double_periodic_exp", "periodic_exp_plus_exp"]:
            # Define wrappers for individual components
            def model_function_wrapper_f1(t, theta):
                _, f1, _ = self.model.model_function(t, theta)
                return f1

            def model_function_wrapper_f2(t, theta):
                _, _, f2 = self.model.model_function(t, theta)
                return f2

            print("Plotting individual model contours...")
            # First component
            plot_contours(
                model_function_wrapper_f1,
                self.time_axis,
                chains,
                axes[1],
                weights=chains.get_weights(),
                colors=self.model_color_map,
            )
            axes[1].set_ylabel("SNR", fontsize=14)
            axes[1].tick_params(axis="both", which="major", labelsize=12)
            axes[1].set_title("First Constituent Model", fontsize=16)

            # Second component
            plot_contours(
                model_function_wrapper_f2,
                self.time_axis,
                chains,
                axes[2],
                weights=chains.get_weights(),
                colors=self.model_color_map,
            )
            axes[2].set_xlabel("Time (s)", fontsize=14)
            axes[2].set_ylabel("SNR", fontsize=14)
            axes[2].tick_params(axis="both", which="major", labelsize=12)
            axes[2].set_title("Second Constituent Model", fontsize=16)
            print("Done!")
        else:
            # Plot lines for non-combined models
            plot_lines(
                model_function_wrapper,
                self.time_axis,
                chains,
                axes[1],
                weights=chains.get_weights(),
                color=self.model_color,
            )
            axes[1].set_xlabel("Time (s)", fontsize=14)
            axes[1].set_ylabel("SNR", fontsize=14)
            axes[1].tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f"{self.file_root}_f_posterior_combined.png")
        )
        plt.close()

    def process_chains(self):
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        chains = read_chains(
            os.path.join(global_settings.get("base_dir"), self.file_root),
            columns=self.paramnames_all,
        )

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
            # Handle combined models separately
            if self.model_name in ["double_periodic_exp", "periodic_exp_plus_exp"]:
                # Collect parameters for plotting
                paramnames_sigma = [r"$\sigma$"]

                if self.model_name == "double_periodic_exp":
                    # First constituent model
                    n1 = self.model.n1
                    n2 = self.model.n2

                    paramnames_A += self.paramnames_all[:n1]
                    paramnames_tau += self.paramnames_all[n1 : 2 * n1]
                    paramnames_u0.append(self.paramnames_all[2 * n1])
                    paramnames_T.append(self.paramnames_all[2 * n1 + 1])

                    # Second constituent model
                    start_idx = 2 * n1 + 2
                    paramnames_A += self.paramnames_all[start_idx : start_idx + n2]
                    paramnames_tau += self.paramnames_all[
                        start_idx + n2 : start_idx + 2 * n2
                    ]
                    paramnames_u0.append(self.paramnames_all[start_idx + 2 * n2])
                    paramnames_T.append(self.paramnames_all[start_idx + 2 * n2 + 1])

                    sigma_index = self.model.get_sigma_param_index()
                    paramnames_sigma = [self.paramnames_all[sigma_index]]

                    if self.fit_pulses:
                        Npulse_indices = self.model.get_Npulse_param_index()
                        paramnames_Npulse = [
                            self.paramnames_all[idx] for idx in Npulse_indices
                        ]

                elif self.model_name == "periodic_exp_plus_exp":
                    # Periodic constituent
                    n1 = self.model.n1
                    n2 = self.model.n2

                    paramnames_A += self.paramnames_all[:n1]
                    paramnames_tau += self.paramnames_all[n1 : 2 * n1]
                    paramnames_u0.append(self.paramnames_all[2 * n1])
                    paramnames_T.append(self.paramnames_all[2 * n1 + 1])

                    # Exponential constituent
                    start_exp_idx = 2 * n1 + 2
                    paramnames_A += self.paramnames_all[
                        start_exp_idx : start_exp_idx + n2
                    ]
                    paramnames_tau += self.paramnames_all[
                        start_exp_idx + n2 : start_exp_idx + 2 * n2
                    ]
                    paramnames_u += self.paramnames_all[
                        start_exp_idx + 2 * n2 : start_exp_idx + 3 * n2
                    ]

                    sigma_index = self.model.get_sigma_param_index()
                    paramnames_sigma = [self.paramnames_all[sigma_index]]

                    if self.fit_pulses:
                        Npulse_indices = self.model.get_Npulse_param_index()
                        paramnames_Npulse = [
                            self.paramnames_all[idx] for idx in Npulse_indices
                        ]
                else:
                    pass  # Other models

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
            else:
                # For other periodic models
                dim = self.model.dim
                paramnames_A = self.paramnames_all[: self.max_peaks]
                paramnames_tau = self.paramnames_all[
                    self.max_peaks : 2 * self.max_peaks
                ]

                u0_indices = self.model.get_u0_param_indices()
                T_indices = self.model.get_period_param_indices()
                sigma_index = self.model.get_sigma_param_index()

                paramnames_u0 = [self.paramnames_all[idx] for idx in u0_indices]
                paramnames_T = [self.paramnames_all[idx] for idx in T_indices]
                paramnames_sigma = [self.paramnames_all[sigma_index]]

                if self.fit_pulses:
                    Npulse_index = self.model.get_Npulse_param_index()
                    paramnames_Npulse = [self.paramnames_all[Npulse_index]]

                if hasattr(self.model, "has_w") and self.model.has_w:
                    paramnames_w = self.paramnames_all[
                        2 * self.max_peaks : 3 * self.max_peaks
                    ]

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
                if paramnames_u0:
                    paramnames_u = (
                        paramnames_u0
                        + paramnames_T
                        + paramnames_sigma
                        + paramnames_Npulse
                    )
                else:
                    paramnames_u = paramnames_sigma + paramnames_Npulse
                if paramnames_w:
                    paramnames_w = paramnames_w + paramnames_sigma + paramnames_Npulse
        else:
            # For non-periodic models
            paramnames_A = self.paramnames_all[: self.max_peaks]
            paramnames_tau = self.paramnames_all[self.max_peaks : 2 * self.max_peaks]
            paramnames_u = self.paramnames_all[2 * self.max_peaks : 3 * self.max_peaks]
            sigma_index = self.model.get_sigma_param_index()
            paramnames_sigma = [self.paramnames_all[sigma_index]]

            if self.fit_pulses:
                Npulse_index = self.model.get_Npulse_param_index()
                paramnames_Npulse = [self.paramnames_all[Npulse_index]]

            if hasattr(self.model, "has_w") and self.model.has_w:
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

        # Create corner plots
        fig, ax = make_2d_axes(paramnames_subset, figsize=(6, 6))
        print("Plot subset...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(
            os.path.join(self.results_dir, f"{self.file_root}_ss_posterior.png")
        )
        plt.close()

        fig, ax = make_2d_axes(paramnames_amp, figsize=(6, 6))
        print("Plot amplitude...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(
            os.path.join(self.results_dir, f"{self.file_root}_amp_posterior.png")
        )
        plt.close()

        fig, ax = make_2d_axes(paramnames_tau, figsize=(6, 6))
        print("Plot tau...")
        chains.plot_2d(ax, color=self.model_color)
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig.savefig(
            os.path.join(self.results_dir, f"{self.file_root}_tau_posterior.png")
        )
        plt.close()

        if paramnames_u:
            fig, ax = make_2d_axes(paramnames_u, figsize=(6, 6))
            print("Plot u...")
            chains.plot_2d(ax, color=self.model_color)
            ax.tick_params(axis="both", which="major", labelsize=12)
            fig.savefig(
                os.path.join(self.results_dir, f"{self.file_root}_u_posterior.png")
            )
            plt.close()

        if paramnames_w:
            fig, ax = make_2d_axes(paramnames_w, figsize=(6, 6))
            print("Plot w...")
            chains.plot_2d(ax, color=self.model_color)
            ax.tick_params(axis="both", which="major", labelsize=12)
            fig.savefig(
                os.path.join(self.results_dir, f"{self.file_root}_w_posterior.png")
            )
            plt.close()

        if self.is_periodic_model:
            self.plot_period_distribution(chains)

    def plot_period_distribution(self, chains):
        if self.is_periodic_model:
            period_indices = self.model.get_period_param_indices()
            for idx, period_param_index in enumerate(period_indices):
                period_param_name = self.paramnames_all[period_param_index]
                period_chain = chains[period_param_name]

                mean_period = period_chain.mean()
                std_period = period_chain.std()

                # Plot using anesthetic
                fig, ax = plt.subplots(figsize=(10, 6))
                hist_plot_1d(
                    ax,
                    period_chain,
                    weights=chains.get_weights(),
                    color=self.model_color,
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

                ax.set_xlabel(f"Period (T{idx+1}) (s)", fontsize=14)
                ax.set_ylabel("Probability Density", fontsize=14)
                ax.tick_params(axis="both", which="major", labelsize=12)
                ax.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        f"{self.file_root}_period_distribution_{idx+1}.png",
                    )
                )
                plt.close()

                print(f"Plotting T{idx+1}...")
        else:
            print("Model is not periodic; period distribution plot not generated.")

    def overlay_predictions_on_input(self):
        chains = read_chains(
            os.path.join(global_settings.get("base_dir"), self.file_root),
            columns=self.paramnames_all,
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1]},
        )

        axes[0].plot(
            self.time_axis,
            self.pulse_profile_snr,
            label="Pulse Profile SNR",
            color="black",
            linewidth=1.5,
            alpha=0.8,
        )

        def model_function_wrapper(t, theta):
            # Check if the model is one of the combined models
            if self.model_name in ["double_periodic_exp", "periodic_exp_plus_exp"]:
                pp_, f1, f2 = self.model.model_function(t, theta)
                return pp_
            else:
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

        print("Overlaying combined predictions on pulse profile SNR...")
        plot_lines(
            model_function_wrapper,
            self.time_axis,
            chains,
            axes[0],
            weights=chains.get_weights(),
            color=self.model_color,
            linewidth=1,
            alpha=0.7,
        )
        axes[0].set_ylabel("Signal / Noise", fontsize=14)
        axes[0].tick_params(axis="both", which="major", labelsize=12)
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title("Combined Model", fontsize=16)

        if self.model_name in ["double_periodic_exp", "periodic_exp_plus_exp"]:
            # Define wrappers for individual components
            def model_function_wrapper_f1(t, theta):
                _, f1, _ = self.model.model_function(t, theta)
                return f1

            def model_function_wrapper_f2(t, theta):
                _, _, f2 = self.model.model_function(t, theta)
                return f2

            print("Overlaying first constituent model predictions...")
            axes[1].plot(
                self.time_axis,
                self.pulse_profile_snr,
                label="Pulse Profile SNR",
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )
            plot_lines(
                model_function_wrapper_f1,
                self.time_axis,
                chains,
                axes[1],
                weights=chains.get_weights(),
                color=self.model_color,
                linewidth=1,
                alpha=0.7,
            )
            axes[1].set_ylabel("Signal / Noise", fontsize=14)
            axes[1].tick_params(axis="both", which="major", labelsize=12)
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_title("First Constituent Model", fontsize=16)

            print("Overlaying second constituent model predictions...")
            axes[2].plot(
                self.time_axis,
                self.pulse_profile_snr,
                label="Pulse Profile SNR",
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )
            plot_lines(
                model_function_wrapper_f2,
                self.time_axis,
                chains,
                axes[2],
                weights=chains.get_weights(),
                color=self.model_color,
                linewidth=1,
                alpha=0.7,
            )
            axes[2].set_xlabel("Time (s)", fontsize=14)
            axes[2].set_ylabel("Signal / Noise", fontsize=14)
            axes[2].tick_params(axis="both", which="major", labelsize=12)
            axes[2].legend()
            axes[2].grid(True)
            axes[2].set_title("Second Constituent Model", fontsize=16)
        else:
            # For other models, plot only the combined prediction
            axes[1].set_visible(False)
            axes[2].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.results_dir, f"{self.file_root}_overlay_predictions_on_snr.png"
            )
        )
        plt.close()

        print("Plotting done!")
