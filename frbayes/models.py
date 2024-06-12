import numpy as np
from scipy.special import erfc


def emg(t, theta, max_peaks):

    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    w = theta[3 * max_peaks : 4 * max_peaks]

    return (
        (A / (2 * tao))
        * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
        * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
    )


def exponential(t, theta, max_peaks):
    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    return A * np.exp((t - u) / tao)
