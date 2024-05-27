import numpy as np
from scipy.special import erfc
from frbayes.utils import load_settings


class Models:
    def __init__(self, settings):
        self.settings = settings

    def emg(self, t, theta):
        Npulse = theta[4 * self.settings["Npulse"]]
        A = theta[0 : self.settings["max_peaks"]]
        tao = theta[self.settings["max_peaks"] : 2 * self.settings["max_peaks"]]
        u = theta[2 * self.settings["max_peaks"] : 3 * self.settings["max_peaks"]]
        w = theta[3 * self.settings["max_peaks"] : 4 * self.settings["max_peaks"]]
        s = np.zeros((self.settings["max_peaks"], len(t)))

        def f(t, A, tao, u, w):
            return (
                (A / (2 * tao))
                * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
                * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
            )

        for i in range(self.settings["max_peaks"]):
            if i < Npulse:
                s[i] = f(t, A[i], tao[i], u[i], w[i])
            else:
                s[i] = 0 * np.ones(len(t))

        return np.sum(s, axis=0)

    def emg_npf(self, t, theta):
        Npulse = self.settings["Npulse"]
        A = theta[0 : self.settings["max_peaks"]]
        tao = theta[self.settings["max_peaks"] : 2 * self.settings["max_peaks"]]
        u = theta[2 * self.settings["max_peaks"] : 3 * self.settings["max_peaks"]]
        w = theta[3 * self.settings["max_peaks"] : 4 * self.settings["max_peaks"]]
        s = np.zeros((self.settings["max_peaks"], len(t)))

        def f(t, A, tao, u, w):
            return (
                (A / (2 * tao))
                * np.exp(((u - t) / tao) + ((2 * w**2) / tao**2))
                * erfc((((u - t) * tao) + w**2) / (w * tao * np.sqrt(2)))
            )

        for i in range(self.settings["max_peaks"]):
            s[i] = f(t, A[i], tao[i], u[i], w[i])

        return np.sum(s, axis=0)
