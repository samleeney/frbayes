import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior
from frbayes.analysis import FRBAnalysis
import yaml
import os
from scipy.special import erfc, logsumexp

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

pp = pp + np.abs(np.min(pp))  # shift to only positive
#
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


# Define the EMG model function
def emg(t, A, tao, u, w):
    return (
        (A / (2 * tao))
        * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
        * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
    )


def logemg(t, A, tao, u, w):
    return (
        np.log((A / (2 * tao)))
        + ((u - t) / tao)
        + ((2 * w**2) / tao**2)
        + np.log(erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2))))
    )


# Define parameters for the EMG function
A = 1  # Amplitude
tao = 2  # scattering timescale
u = 1  # Mean of the Gaussian component
w = 0.1  # width
sigma = 1  # Standard deviation of the Gaussian component


# Generate a range of t values
t = np.linspace(-5, 5, 1000)

# Compute the EMG function values
emg_values = np.exp(logemg(t, A, tao, u, w))

# Add noise to the EMG function values
noise_level = 0.1
noise = np.random.normal(0, noise_level, t.shape)
pp = emg_values + noise

# # # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(t, emg_values, label="EMG", color="blue")
# plt.plot(t, pp, label="EMG with Noise", color="red")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel("Value")
# plt.title("Exponentially Modified Gaussian with Noise")
# plt.show()
# jk


# Define the Gaussian model likelihood
def loglikelihood(theta):
    """Gaussian Model Likelihood"""
    maxNpulse = 10
    Npulse = theta[4 * maxNpulse]
    sigma = theta[(4 * maxNpulse) + 1]
    A = theta[0:maxNpulse]
    tao = theta[maxNpulse : 2 * maxNpulse]
    u = theta[2 * maxNpulse : 3 * maxNpulse]
    w = theta[3 * maxNpulse : 4 * maxNpulse]
    # print(A, tao, u, w, sigma, Npulse)
    # sigma_pulse = theta[4 * maxNpulse : 5 * maxNpulse]

    # Assuming t and pp are globally defined
    s = np.zeros((10, len(t)))

    for i in range(maxNpulse):
        if i < Npulse:
            s[i] = logemg(t, A[i], tao[i], u[i], w[i])  # , sigma_pulse[i])
        else:
            s[i] = -np.inf

    # print(s)

    model = logsumexp(s, axis=0)

    logL = (
        np.log(1 / (sigma * np.sqrt(2 * np.pi)))
        - 0.5 * ((pp - np.exp(model)) ** 2) / (sigma**2)
    ).sum()

    return logL, []


def prior(hypercube):
    N = 10

    theta = np.zeros_like(hypercube)

    # Populate each parameter array
    for i in range(N):
        theta[i] = UniformPrior(0.001, 2)(hypercube[i])  # A
        theta[N + i] = UniformPrior(0.001, 3)(hypercube[N + i])  # tao
        theta[(2 * N) + i] = UniformPrior(0.001, 2)(hypercube[(2 * N) + i])  # u
        theta[(3 * N) + i] = UniformPrior(0.001, 1)(hypercube[(3 * N) + i])  # w
        # theta[4 * N + i] = UniformPrior(0, 1)(hypercube[4 * N + i])  # sigma_pulse
    theta[4 * N] = UniformPrior(0, 3)(hypercube[4 * N])  # Npulse
    theta[(4 * N) + 1] = UniformPrior(0.001, 1)(hypercube[(4 * N) + 1])  # sigma

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
        nlive=100,
        do_clustering=True,
        read_resume=False,
    )


if __name__ == "__main__":
    run_polychord(file_root)
