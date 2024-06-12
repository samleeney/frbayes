import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior, LogUniformPrior
from frbayes.analysis import FRBAnalysis
import yaml
import os
from scipy.special import erfc
from frbayes.models import emg, exponential
from frbayes.settings import global_settings

try:
    from mpi4py import MPI
except ImportError:
    pass


class FRBModel:
    def __init__(self):
        self.analysis = FRBAnalysis()
        self.pp = self.analysis.pulse_profile_snr
        self.t = self.analysis.time_axis
        self.max_peaks = global_settings.get("max_peaks")
        self.pp += np.abs(np.min(self.pp))  # shift to only positive
        print(global_settings.get("model"))
        if global_settings.get("model") == "emg":
            self.model = emg
        elif global_settings.get("model") == "exponential":
            self.model = exponential

        if global_settings.get("model") == "emg":
            self.nDims = (self.max_peaks * 4) + 1
        elif global_settings.get("model") == "exponential":
            self.nDims = (self.max_peaks * 3) + 1

        if global_settings.get("fit_pulses"):
            self.nDims += 1

    def loglikelihood(self, theta):
        """Gaussian Model Likelihood"""

        # Sigma location in array depends on model
        if global_settings.get("model") == "emg":
            sigma = theta[(4 * self.max_peaks)]
        elif global_settings.get("model") == "exponential":
            sigma = theta[(3 * self.max_peaks)]

        # Check if fitting for npulse or not. If not, set to max_peaks.
        if global_settings.get("fit_pulses"):
            Npulse = theta[(4 * self.max_peaks) + 1]
        else:
            Npulse = self.max_peaks

        s = np.zeros((self.max_peaks, len(self.t)))

        for i in range(self.max_peaks):
            if i < Npulse:
                s[i] = self.model(self.t, theta, self.max_peaks, i)
            else:
                s[i] = np.zeros(len(self.t))

        pp_ = np.sum(s, axis=0)
        logL = (
            np.log(1 / (sigma * np.sqrt(2 * np.pi)))
            - 0.5 * ((self.pp - pp_) ** 2) / (sigma**2)
        ).sum()

        return logL, []

    def prior(self, hypercube):
        theta = np.zeros_like(hypercube)

        for i in range(self.max_peaks):
            theta[i] = UniformPrior(0, 5)(hypercube[i])  # Amplitude A
            theta[self.max_peaks + i] = UniformPrior(1, 5)(
                hypercube[self.max_peaks + i]
            )  # Time constant tao
            theta[2 * self.max_peaks + i] = UniformPrior(0, 5)(
                hypercube[2 * self.max_peaks + i]
            )  # Location u
            if global_settings.get("model") == "emg":
                theta[3 * self.max_peaks + i] = UniformPrior(0, 5)(
                    hypercube[3 * self.max_peaks + i]
                )  # Width w

        # The number of parameters in the model changes depending on model. This accoutns for that.
        if global_settings.get("model") == "emg":
            dim = 4
        elif global_settings.get("model") == "exponential":
            dim = 3

        theta[dim * self.max_peaks] = LogUniformPrior(0.001, 1)(
            hypercube[dim * self.max_peaks]
        )  # Noise sigma

        if global_settings.get("fit_pulses"):
            theta[dim * self.max_peaks + 1] = UniformPrior(1, self.max_peaks)(
                hypercube[dim * self.max_peaks + 1]
            )  # Number of pulses Npulse

        return theta

    def run_polychord(self):
        nDerived = 0

        output = pypolychord.run(
            self.loglikelihood,
            self.nDims,
            nlive=10,
            nDerived=nDerived,
            prior=self.prior,
            file_root=global_settings.get("file_root"),
            do_clustering=True,
            read_resume=True,
        )


if __name__ == "__main__":
    frb_model = FRBModel()
    frb_model.run_polychord()
