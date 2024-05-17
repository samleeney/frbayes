import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior
from frbayes.analysis import FRBAnalysis
import yaml
import os
from scipy.special import erfc

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


# Define the EMG model function
def emg(t, A, lambda_, u, sigma):
    # print(lambda_ / 2)
    # print("--")
    # print((2 * u + lambda_ * sigma**2 - 2 * t))
    # print("--")
    # print(np.exp((lambda_ / 2) * (2 * u + lambda_ * sigma**2 - 2 * t)))
    # print(erfc((u + lambda_ * sigma**2 - t) / (np.sqrt(2) * sigma)))

    return A * (
        (lambda_ / 2)
        * np.exp((lambda_ / 2) * (2 * u + lambda_ * sigma**2 - 2 * t))
        * erfc((u + lambda_ * sigma**2 - t) / (np.sqrt(2) * sigma))
    )


# Define the Gaussian model likelihood
def loglikelihood(theta):
    """Gaussian Model Likelihood"""
    maxNpulse = 10
    Npulse = theta[4 * maxNpulse]
    sigma = theta[4 * maxNpulse + 1]
    A = theta[0:maxNpulse]
    lambda_ = theta[maxNpulse : 2 * maxNpulse]
    u = theta[2 * maxNpulse : 3 * maxNpulse]
    sigma_pulse = theta[3 * maxNpulse : 4 * maxNpulse]

    # Assuming t and pp are globally defined
    s = np.zeros((10, len(t)))

    for i in range(1, maxNpulse + 1):
        if i <= Npulse:
            s[i - 1] = emg(t, A[i - 1], lambda_[i - 1], u[i - 1], sigma_pulse[i - 1])
        else:
            s[i - 1] = 0

    # print(s)

    model = s.sum(axis=0)

    logL = (
        np.log(1 / (sigma * np.sqrt(2 * np.pi)))
        - 0.5 * ((pp - model) ** 2) / (sigma**2)
    ).sum()

    return logL, []


def prior(hypercube):
    N = 10

    theta = np.zeros_like(hypercube)

    # Populate each parameter array
    for i in range(N):
        theta[i] = UniformPrior(0, 100)(hypercube[i])  # A
        theta[N + i] = UniformPrior(0, 10)(hypercube[N + i])  # lambda
        theta[2 * N + i] = UniformPrior(0.001, 5)(hypercube[2 * N + i])  # u
        theta[3 * N + i] = UniformPrior(0, 1)(hypercube[3 * N + i])  # sigma_pulse
    theta[4 * N] = np.round(UniformPrior(1, 20)(hypercube[4 * N]))  # Npulse
    theta[4 * N + 1] = UniformPrior(0.001, 5)(hypercube[4 * N + 1])  # sigma

    return theta


# Run PolyChord with the Gaussian model
def run_polychord(file_root):
    nDims = 42  # Amplitude, center, width
    nDerived = 0

    output = pypolychord.run(
        loglikelihood,
        nDims,
        nDerived=nDerived,
        prior=prior,
        file_root=file_root,
        nlive=200,
        do_clustering=True,
        read_resume=False,
    )


if __name__ == "__main__":
    run_polychord(file_root)
