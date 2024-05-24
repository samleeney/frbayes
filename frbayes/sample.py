import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior, LogUniformPrior
from frbayes.analysis import FRBAnalysis
import yaml
import os
from scipy.special import erfc
from frbayes.models import emg

try:
    from mpi4py import MPI
except ImportError:
    pass


# Load settings
def load_settings(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


# Load preprocessed data
settings = load_settings("settings.yaml")
analysis = FRBAnalysis(settings)
pp = analysis.pulse_profile_snr
t = analysis.time_axis
max_peaks = settings["max_peaks"]
pp = pp + np.abs(np.min(pp))  # shift to only positive


# Define the Gaussian model likelihood
def loglikelihood(theta):
    """Gaussian Model Likelihood"""
    Npulse = theta[4 * max_peaks]
    sigma = theta[(4 * max_peaks) + 1]
    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    w = theta[3 * max_peaks : 4 * max_peaks]
    # print(A, tao, u, w, sigma, Npulse)
    # sigma_pulse = theta[4 * maxNpulse : 5 * maxNpulse]

    # Assuming t and pp are globally defined
    s = np.zeros((max_peaks, len(t)))

    for i in range(max_peaks):
        if i < Npulse:

            s[i] = emg(t, A[i], tao[i], u[i], w[i])  # , sigma_pulse[i])
        else:
            s[i] = 0 * np.ones(len(t))

    # print(s)

    model = np.sum(s, axis=0)

    logL = (
        np.log(1 / (sigma * np.sqrt(2 * np.pi)))
        - 0.5 * ((pp - model) ** 2) / (sigma**2)
    ).sum()

    return logL, []


def prior(hypercube):

    theta = np.zeros_like(hypercube)

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
        # theta[4 * N + i] = UniformPrior(0, 1)(hypercube[4 * N + i])  # sigma_pulse
    theta[4 * max_peaks] = UniformPrior(1, max_peaks)(
        hypercube[4 * max_peaks]
    )  # Npulse
    theta[(4 * max_peaks) + 1] = LogUniformPrior(0.001, 1)(
        hypercube[(4 * max_peaks) + 1]
    )  # sigma

    return theta


# Run PolyChord with the Gaussian model
def run_polychord(file_root):
    nDims = max_peaks * 4 + 2  # Amplitude, center, width
    nDerived = 0

    output = pypolychord.run(
        loglikelihood,
        nDims,
        nDerived=nDerived,
        prior=prior,
        file_root=file_root,
        do_clustering=True,
        read_resume=True,
    )


if __name__ == "__main__":
    run_polychord(file_root)