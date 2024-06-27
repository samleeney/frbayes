import numpy as np
from scipy.special import erfc


def emg(t, theta, max_peaks, i):

    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    w = theta[3 * max_peaks : 4 * max_peaks]

    return (
        (A[i] / (2 * tao[i]))
        * np.exp(((u[i] - t) / tao[i]) + ((2 * w[i] ** 2) / tao[i] ** 2))
        * erfc((((u[i] - t) * tao[i]) + w[i] ** 2) / (w[i] * tao[i] * np.sqrt(2)))
    )


def exponential(t, theta, max_peaks, i):
    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    f = A[i] * np.exp(-(t - u[i]) / tao[i])
    f = np.where(t <= u[i], 0, f)
    return f
