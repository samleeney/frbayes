import numpy as np
from scipy.special import erfc
from frbayes.utils import load_settings

settings = load_settings()


def emg(t, theta):
    max_peaks = settings["max_peaks"]
    Npulse = theta[4 * max_peaks]
    A = theta[0:max_peaks]
    tao = theta[max_peaks : 2 * max_peaks]
    u = theta[2 * max_peaks : 3 * max_peaks]
    w = theta[3 * max_peaks : 4 * max_peaks]
    s = np.zeros((max_peaks, len(t)))

    def f(t, A, tao, u, w):
        return (
            (A / (2 * tao))
            * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
            * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
        )

    for i in range(max_peaks):
        if i < Npulse:
            s[i] = f(t, A[i], tao[i], u[i], w[i])
        else:
            s[i] = 0 * np.ones(len(t))

    return np.sum(s, axis=0)
