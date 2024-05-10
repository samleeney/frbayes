from numpy import pi, exp, log
import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior
import os

try:
    from mpi4py import MPI
except ImportError:
    pass

# Load preprocessed data
pulse_profile_snr = np.loadtxt("results/pulse_profile_snr.csv", delimiter=",")
time_axis = np.loadtxt("results/time_axis.csv", delimiter=",")
data = {"pulse_profile_snr": pulse_profile_snr, "time_axis": time_axis}


# Define the repeating Gaussian model likelihood
def repeating_gaussian_likelihood(theta):
    """Repeating Gaussian Model Likelihood"""
    nGauss = len(theta) // 3
    amplitude, center, width = (
        theta[:nGauss],
        theta[nGauss : 2 * nGauss],
        theta[2 * nGauss :],
    )
    model = sum(
        a * exp(-((data["time_axis"] - c) ** 2) / (2 * w**2))
        for a, c, w in zip(amplitude, center, width)
    )
    residuals = data["pulse_profile_snr"] - model
    logL = -0.5 * np.sum(residuals**2)
    return logL, []


# Define a uniform prior
def repeating_gaussian_prior(hypercube):
    nGauss = len(hypercube) // 3
    amplitude = [UniformPrior(0, 10)(hypercube[i]) for i in range(nGauss)]
    center = [
        UniformPrior(min(data["time_axis"]), max(data["time_axis"]))(
            hypercube[i + nGauss]
        )
        for i in range(nGauss)
    ]
    width = [UniformPrior(0.001, 0.1)(hypercube[i + 2 * nGauss]) for i in range(nGauss)]
    return amplitude + center + width


# Run PolyChord with the repeating Gaussian model
def run_polychord():
    nGauss = 3
    nDims = nGauss * 3
    nDerived = 0
    paramnames = (
        [(f"amplitude_{i}", f"A_{i}") for i in range(nGauss)]
        + [(f"center_{i}", f"c_{i}") for i in range(nGauss)]
        + [(f"width_{i}", f"w_{i}") for i in range(nGauss)]
    )

    output = pypolychord.run(
        repeating_gaussian_likelihood,
        nDims,
        nDerived=nDerived,
        prior=repeating_gaussian_prior,
        file_root="repeating_gaussian",
        nlive=100,
        do_clustering=True,
        read_resume=False,
        paramnames=paramnames,
    )


run_polychord()
