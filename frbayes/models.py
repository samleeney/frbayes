import numpy as np
from scipy.special import erfc


def emg(t, A, tao, u, w):
    return (
        (A / (2 * tao))
        * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
        * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
    )


def exponential(t, A, tao, u, w=None):
    return A * np.exp((t - u) / tao)
