import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior, LogUniformPrior
from frbayes.analysis import FRBAnalysis
from frbayes.models import Models
from scipy.special import erfc

try:
    from mpi4py import MPI
except ImportError:
    pass


class Sample:
    def __init__(self, settings):
        self.settings = settings

        # Load preprocessed data
        self.analysis = FRBAnalysis(self.settings)
        self.pp = self.analysis.pulse_profile_snr
        self.t = self.analysis.time_axis
        self.max_peaks = self.settings["max_peaks"]
        self.pp = self.pp + np.abs(np.min(self.pp))  # shift to only positive

        if self.settings["Npulse"] == "free":
            self.emg_model = Models(self.settings).emg
        else:
            self.emg_model = Models(self.settings).emg_npf

    # Define the Gaussian model likelihood
    def loglikelihood(self, theta):
        """Gaussian Model Likelihood"""
        model = self.emg_model(self.t, theta)
        sigma = theta[(4 * self.settings["max_peaks"])]

        logL = (
            np.log(1 / (sigma * np.sqrt(2 * np.pi)))
            - 0.5 * ((self.pp - model) ** 2) / (sigma**2)
        ).sum()

        return logL, []

    def prior(self, hypercube):
        theta = np.zeros_like(hypercube)

        # Populate each parameter array
        # Populate each parameter array

        for i in range(self.settings["max_peaks"]):
            theta[i] = UniformPrior(0, 5)(hypercube[i])  # A
            theta[self.settings["max_peaks"] + i] = UniformPrior(1, 5)(
                hypercube[self.settings["max_peaks"] + i]
            )  # tao (keep greater than 1 to avoid overflow)
            theta[(2 * self.settings["max_peaks"]) + i] = UniformPrior(0, 5)(
                hypercube[(2 * self.settings["max_peaks"]) + i]
            )  # u
            theta[(3 * self.settings["max_peaks"]) + i] = UniformPrior(0, 5)(
                hypercube[(3 * self.settings["max_peaks"]) + i]
            )  # w

        theta[(4 * self.settings["max_peaks"])] = LogUniformPrior(0.001, 1)(
            hypercube[(4 * self.settings["max_peaks"])]
        )  # sigma
        if self.settings["Npulse"] == "free":
            theta[4 * self.settings["max_peaks"] + 1] = UniformPrior(
                1, self.settings["max_peaks"]
            )(
                hypercube[4 * self.settings["max_peaks"] + 1]
            )  # Npulse

        return theta

    # Run PolyChord with the Gaussian model
    def run_polychord(self):
        if self.settings["Npulse"] == "free":
            nDims = self.settings["max_peaks"] * 4 + 2  # Amplitude, center, width
            print("Npulse is a free parameter")
        else:
            self.settings["max_peaks"] = self.settings["Npulse"]
            nDims = self.settings["max_peaks"] * 4 + 1
            print("Npulse is fixed to " + str(self.settings["Npulse"]))

        nDerived = 0

        output = pypolychord.run(
            self.loglikelihood,
            nDims,
            nDerived=nDerived,
            prior=self.prior,
            file_root=self.settings["file_root"],
            do_clustering=True,
            read_resume=True,
        )
        return output
