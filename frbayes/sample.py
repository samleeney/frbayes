import numpy as np
import pypolychord
from pypolychord.priors import UniformPrior, LogUniformPrior
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
max_peaks = settings["max_peaks"]
pp = pp + np.abs(np.min(pp))  # shift to only positive


# Define the EMG model function
def emg(t, A, tao, u, w):
    return (
        (A / (2 * tao))
        * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
        * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
    )


#
# import numpy as np
# import matplotlib.pyplot as plt


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import erfc


# def logemg(t, A, tao, u, w):
#     return (
#         np.log((A / (2 * tao)))
#         + ((u - t) / tao)
#         + ((2 * w**2) / tao**2)
#         + np.log(erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2))))
#     )


# # Define parameters for the EMG function
# A = 1  # Amplitude
# tao = 2  # scattering timescale
# u = 1  # Mean of the Gaussian component
# w = 0.1  # width


# # Generate a range of t values
# t = np.linspace(0, 5, 1000)

# # Compute the EMG function values
# emg_values = emg(t, A, tao, u, w)

# # Add noise to the EMG function values
# noise_level = 0.01
# noise = np.random.normal(0, noise_level, t.shape)
# pp1 = emg_values + noise


# # Define parameters for the EMG function
# A = 1.5  # Amplitude
# tao = 1.8  # scattering timescale
# u = 3  # Mean of the Gaussian component
# w = 0.2  # width
# #


# # Generate a range of t values
# t = np.linspace(0, 5, 1000)

# # Compute the EMG function values
# emg_values = emg(t, A, tao, u, w)

# # Add noise to the EMG function values
# noise_level = 0.01
# noise = np.random.normal(0, noise_level, t.shape)
# pp = pp1 + emg_values + noise

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
    # Npulse = theta[4 * max_peaks]
    sigma = theta[(4 * max_peaks)]
    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    w = theta[3 * max_peaks : 4 * max_peaks]
    # print(A, tao, u, w, sigma, Npulse)
    # sigma_pulse = theta[4 * maxNpulse : 5 * maxNpulse]

    # Assuming t and pp are globally defined
    s = np.zeros((max_peaks, len(t)))

    for i in range(max_peaks):
        s[i] = emg(t, A[i], tao[i], u[i], w[i])  # , sigma_pulse[i])
    # #
    # for i in range(max_peaks):
    #     if i < Npulse:
    #         s[i] = emg(t, A[i], tao[i], u[i], w[i])  # , sigma_pulse[i])
    #     else:
    #         s[i] = 0 * np.ones(len(t))

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

    # theta[4 * max_peaks] = UniformPrior(1, max_peaks)(hypercube[4 * max_peaks]) #
    # )  # Npulse
    theta[(4 * max_peaks)] = LogUniformPrior(0.001, 1)(
        hypercube[(4 * max_peaks)]
    )  # sigma

    return theta


# Run PolyChord with the Gaussian model
def run_polychord(file_root):
    nDims = max_peaks * 4 + 1  # Amplitude, center, width
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
