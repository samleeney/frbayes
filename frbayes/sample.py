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
        self.sigma = None
        if global_settings.get("model") == "emg":
            self.model = emg
        elif global_settings.get("model") == "exponential":
            self.model = exponential

    def loglikelihood(self, theta):
        """Gaussian Model Likelihood"""
        sigma = theta[(4 * self.max_peaks)]
        self.sigma = sigma

        A = theta[0 : self.max_peaks]
        tao = theta[self.max_peaks : 2 * self.max_peaks]
        u = theta[2 * self.max_peaks : 3 * self.max_peaks]

        if global_settings.get("model") == "emg":
            w = theta[3 * self.max_peaks : 4 * self.max_peaks]

        if global_settings.get("fit_pulses"):
            if global_settings.get("model") == "emg":
                Npulse = theta[(4 * self.max_peaks) + 1]
            elif global_settings.get("model") == "exponential":
                Npulse = theta[(3 * self.max_peaks) + 1]

        else:
            Npulse = self.max_peaks

        s = np.zeros((self.max_peaks, len(self.t)))

        for i in range(self.max_peaks):
            if i < Npulse:
                s[i] = self.model(self.t, A[i], tao[i], u[i], w[i])
            else:
                s[i] = np.zeros(len(self.t))

        model = np.sum(s, axis=0)
        logL = (
            np.log(1 / (sigma * np.sqrt(2 * np.pi)))
            - 0.5 * ((self.pp - model) ** 2) / (sigma**2)
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
            theta[3 * self.max_peaks + i] = UniformPrior(0, 5)(
                hypercube[3 * self.max_peaks + i]
            )  # Width w
        theta[4 * self.max_peaks] = LogUniformPrior(0.001, 1)(
            hypercube[4 * self.max_peaks]
        )  # Noise sigma

        if global_settings.get("fit_pulses"):
            theta[4 * self.max_peaks + 1] = UniformPrior(1, self.max_peaks)(
                hypercube[4 * self.max_peaks + 1]
            )  # Number of pulses Npulse

        return theta

    def run_polychord(self):
        nDims = self.max_peaks * 4 + 1
        if global_settings.get("fit_pulses"):
            nDims += 1

        nDerived = 0

        output = pypolychord.run(
            self.loglikelihood,
            nDims,
            nDerived=nDerived,
            prior=self.prior,
            file_root=global_settings.get("file_root"),
            do_clustering=True,
            read_resume=True,
        )


if __name__ == "__main__":
    frb_model = FRBModel()
    frb_model.run_polychord()
