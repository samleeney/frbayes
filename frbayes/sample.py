import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior, LogUniformPrior
from frbayes.analysis import FRBAnalysis
from frbayes.models import Models
from scipy.special import erfc
from frbayes.models import emg
from frbayes.settings import global_settings

try:
    from mpi4py import MPI
except ImportError:
    pass

# Load preprocessed data
analysis = FRBAnalysis()
pp = analysis.pulse_profile_snr
t = analysis.time_axis
max_peaks = global_settings.get("max_peaks")
pp = pp + np.abs(np.min(pp))  # shift to only positive


# Define the Gaussian model likelihood
def loglikelihood(theta):
    """Gaussian Model Likelihood"""
    sigma = theta[(4 * max_peaks)]

    if global_settings.get("fit_pulses") is True:
        Npulse = theta[(4 * max_peaks) + 1]
    else:
        Npulse = max_peaks

    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    w = theta[3 * max_peaks : 4 * max_peaks]

    # Assuming t and pp are globally defined
    s = np.zeros((max_peaks, len(t)))

    for i in range(max_peaks):
        if i < Npulse:

            s[i] = emg(t, A[i], tao[i], u[i], w[i])  # , sigma_pulse[i])
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

    # Populate each parameter array
    for i in range(max_peaks):
        theta[i] = UniformPrior(0, 5)(hypercube[i])  # A
        theta[max_peaks + i] = UniformPrior(1, 5)(
            hypercube[max_peaks + i]
        )  # tao (keep greater than 1 to avoid overflow)
        theta[(2 * max_peaks) + i] = UniformPrior(0, 5)(
            hypercube[(2 * max_peaks) + i]
        )  # u
        theta[(3 * max_peaks) + i] = UniformPrior(0, 5)(
            hypercube[(3 * max_peaks) + i]
        )  # w
    theta[(4 * max_peaks)] = LogUniformPrior(0.001, 1)(
        hypercube[(4 * max_peaks)]
    )  # sigma

    if global_settings.get("fit_pulses") is True:
        theta[(4 * max_peaks) + 1] = UniformPrior(1, max_peaks)(
            hypercube[(4 * max_peaks) + 1]
        )  # Npulse

    return theta

        nDerived = 0

# Run PolyChord with the Gaussian model
def run_polychord(file_root):

    if global_settings.get("fit_pulses") is True:
        nDims = max_peaks * 4 + 2
    else:
        nDims = max_peaks * 4 + 1

    nDerived = 0

    output = pypolychord.run(
        loglikelihood,
        nDims,
        nDerived=nDerived,
        prior=prior,
        file_root=file_root,
        do_clustering=True,
        read_resume=False,
    )


if __name__ == "__main__":
    run_polychord(file_root)
