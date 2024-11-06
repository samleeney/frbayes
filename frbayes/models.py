import numpy as np
from scipy.special import erfc
from pypolychord.priors import UniformPrior, LogUniformPrior, SortedUniformPrior
from frbayes.settings import global_settings


class BaseModel:
    """
    Base class for FRB models.

    Attributes:
        settings (Settings): Global settings instance.
        max_peaks (int): Maximum number of peaks.
        fit_pulses (bool): Whether to fit the number of pulses.
        nDims (int): Number of dimensions/parameters in the model.
        dim (int): Number of parameters per peak.
        color (str): Color associated with the model for plotting.
        paramnames_all (list): List of parameter names in LaTeX format.
    """

    def __init__(self, settings):
        self.settings = settings
        self.max_peaks = self.settings.get("max_peaks")
        self.fit_pulses = self.settings.get("fit_pulses")
        self.nDims = None  # To be defined in subclass
        self.dim = None  # Dimensionality per peak, to be defined in subclass
        self.color = "black"  # Default plotting color
        self.paramnames_all = []  # To be defined in subclass

    def model_function(self, t, theta, i):
        """
        Compute the model function for a given peak.

        Args:
            t (array): Time axis.
            theta (array): Model parameters.
            i (int): Index of the peak.

        Returns:
            array: Model function evaluated at time t for peak i.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood given theta and data.

        Args:
            theta (array): Model parameters.
            data (dict): Dictionary containing 'pp' (pulse profile) and 't' (time axis).

        Returns:
            tuple: (log-likelihood value, empty list)
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def prior(self, hypercube):
        """
        Transform unit hypercube samples to the parameter space.

        Args:
            hypercube (array): Samples from the unit hypercube.

        Returns:
            array: Transformed samples in the parameter space.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_period_param_indices(self):
        """
        Returns the indices of period parameters in theta.
        """
        return []

    def get_u0_param_indices(self):
        """
        Returns the indices of u0 parameters in theta.
        """
        return []

    def get_sigma_param_index(self):
        """
        Returns the index of sigma parameter in theta.
        """
        return None

    @property
    def has_w(self):
        """
        Indicates whether the model includes width parameters (w_i).
        """
        return False


class EMGModel(BaseModel):
    """
    Exponentially Modified Gaussian (EMG) Model.

    Inherits from BaseModel and implements the EMG model specifics.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 4  # Number of parameters per peak for EMG model
        self.color = "blue"  # Assign a unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        """
        Setup model parameters and names.
        """
        self.nDims = self.max_peaks * self.dim + 1  # +1 for sigma
        if self.fit_pulses:
            self.nDims += 1  # +1 for Npulse

        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$A_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$\tau_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$u_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$w_{{{}}}$".format(i + 1))

        self.paramnames_all.append(r"$\sigma$")

        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse}}$")

    def model_function(self, t, theta, i):
        """
        Compute the EMG model function for peak i.

        Args:
            t (array): Time axis.
            theta (array): Model parameters.
            i (int): Index of the peak.

        Returns:
            array: Model function evaluated at time t for peak i.
        """
        A = theta[0 : self.max_peaks]
        tau = theta[self.max_peaks : 2 * self.max_peaks]
        u = theta[2 * self.max_peaks : 3 * self.max_peaks]
        w = theta[3 * self.max_peaks : 4 * self.max_peaks]

        return (
            (A[i] / (2 * tau[i]))
            * np.exp(((u[i] - t) / tau[i]) + ((2 * w[i] ** 2) / tau[i] ** 2))
            * erfc((((u[i] - t) * tau[i]) + w[i] ** 2) / (w[i] * tau[i] * np.sqrt(2)))
        )

    @property
    def has_w(self):
        return True

    def prior(self, hypercube):
        """
        Transform unit hypercube samples to the parameter space for the EMG model.
        """
        theta = np.zeros(self.nDims)

        uniform_prior = UniformPrior(0.001, 1.0)
        theta[: self.max_peaks] = uniform_prior(hypercube[: self.max_peaks])  # A_i
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )  # tau_i
        sorted_uniform_prior = SortedUniformPrior(0.001, 4.0)
        theta[2 * self.max_peaks : 3 * self.max_peaks] = sorted_uniform_prior(
            hypercube[2 * self.max_peaks : 3 * self.max_peaks]
        )  # u_i
        theta[3 * self.max_peaks : 4 * self.max_peaks] = uniform_prior(
            hypercube[3 * self.max_peaks : 4 * self.max_peaks]
        )  # w_i
        log_uniform_prior = LogUniformPrior(0.001, 1.0)
        sigma_index = 4 * self.max_peaks
        theta[sigma_index] = log_uniform_prior(hypercube[sigma_index])  # sigma

        if self.fit_pulses:
            Npulse_index = sigma_index + 1
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            theta[Npulse_index] = int(Npulse_prior(hypercube[Npulse_index]))

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the EMG model.
        """
        pp = data["pp"]
        t = data["t"]

        sigma = theta[4 * self.max_peaks]
        if self.fit_pulses:
            Npulse = int(theta[4 * self.max_peaks + 1])
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        for i in range(Npulse):
            pp_ += self.model_function(t, theta, i)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def get_sigma_param_index(self):
        index = 4 * self.max_peaks
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return 4 * self.max_peaks + 1
        else:
            return None


class ExponentialModel(BaseModel):
    """
    Exponential Model.

    Inherits from BaseModel and implements the exponential model specifics.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 3  # Number of parameters per peak for exponential model
        self.color = "green"  # Assign a unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        """
        Setup model parameters and names.
        """
        self.nDims = self.max_peaks * self.dim + 1  # +1 for sigma
        if self.fit_pulses:
            self.nDims += 1  # +1 for Npulse

        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$A_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$\tau_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$u_{{{}}}$".format(i + 1))

        self.paramnames_all.append(r"$\sigma$")
        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse}}$")

    def model_function(self, t, theta, i):
        """
        Compute the exponential model function for peak i.
        """
        A = theta[0 : self.max_peaks]
        tau = theta[self.max_peaks : 2 * self.max_peaks]
        u = theta[2 * self.max_peaks : 3 * self.max_peaks]
        f = A[i] * np.exp(-(t - u[i]) / tau[i])
        f = np.where(t <= u[i], 0.0, f)
        return f

    def prior(self, hypercube):
        """
        Transform unit hypercube samples to the parameter space for the exponential model.
        """
        theta = np.zeros(self.nDims)

        uniform_prior = UniformPrior(0.001, 1.0)
        theta[: self.max_peaks] = uniform_prior(hypercube[: self.max_peaks])  # A_i
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )  # tau_i
        sorted_uniform_prior = SortedUniformPrior(0.001, 4.0)
        theta[2 * self.max_peaks : 3 * self.max_peaks] = sorted_uniform_prior(
            hypercube[2 * self.max_peaks : 3 * self.max_peaks]
        )  # u_i
        log_uniform_prior = LogUniformPrior(0.001, 1.0)
        sigma_index = 3 * self.max_peaks
        theta[sigma_index] = log_uniform_prior(hypercube[sigma_index])  # sigma

        if self.fit_pulses:
            Npulse_index = sigma_index + 1
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            theta[Npulse_index] = Npulse_prior(hypercube[Npulse_index])

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the exponential model.
        """
        pp = data["pp"]
        t = data["t"]

        sigma = theta[3 * self.max_peaks]
        if self.fit_pulses:
            Npulse = int(theta[3 * self.max_peaks + 1])
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        for i in range(Npulse):
            pp_ += self.model_function(t, theta, i)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def get_sigma_param_index(self):
        index = 3 * self.max_peaks
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return 3 * self.max_peaks + 1
        else:
            return None


class PeriodicExponentialModel(BaseModel):
    """
    Periodic Exponential Model.

    Similar to the ExponentialModel, but with a single period parameter controlling all peak locations.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 2  # 2 params per peak
        self.color = "red"  # Assign a unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        self.nDims = self.max_peaks * self.dim + 3  # A*npeak, tau*npeak, u0, T, sigma
        if self.fit_pulses:
            self.nDims += 1  # +1 for Npulse

        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$A_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$\tau_{{{}}}$".format(i + 1))

        self.paramnames_all.append(r"$u_0$")
        self.paramnames_all.append(r"$T$")
        self.paramnames_all.append(r"$\sigma$")
        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse}}$")

    def model_function(self, t, theta, n):
        A = theta[0 : self.max_peaks]
        tau = theta[self.max_peaks : 2 * self.max_peaks]
        u0 = theta[2 * self.max_peaks]
        T = theta[2 * self.max_peaks + 1]

        u_n = u0 + n * T
        f = A[n] * np.exp(-(t - u_n) / tau[n])
        f = np.where(t <= u_n, 0.0, f)
        return f

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)

        uniform_prior = UniformPrior(0.001, 1.0)
        theta[: self.max_peaks] = uniform_prior(hypercube[: self.max_peaks])  # A_i
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )  # tau_i

        uniform_prior_u0 = UniformPrior(0.001, 4.0 / self.max_peaks)
        theta[2 * self.max_peaks] = uniform_prior_u0(
            hypercube[2 * self.max_peaks]
        )  # u0

        uniform_prior_T = UniformPrior(0.001, 4.0 / self.max_peaks)
        theta[2 * self.max_peaks + 1] = uniform_prior_T(
            hypercube[2 * self.max_peaks + 1]
        )  # T

        log_uniform_prior = LogUniformPrior(0.001, 1.0)
        sigma_index = 2 * self.max_peaks + 2
        theta[sigma_index] = log_uniform_prior(hypercube[sigma_index])  # sigma

        if self.fit_pulses:
            Npulse_index = sigma_index + 1
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            theta[Npulse_index] = Npulse_prior(hypercube[Npulse_index])

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the periodic exponential model.
        """
        pp = data["pp"]
        t = data["t"]

        sigma = theta[2 * self.max_peaks + 2]
        if self.fit_pulses:
            Npulse = int(theta[2 * self.max_peaks + 3])
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        for n in range(Npulse):
            pp_ += self.model_function(t, theta, n)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def get_period_param_indices(self):
        return [2 * self.max_peaks + 1]

    def get_u0_param_indices(self):
        return [2 * self.max_peaks]

    def get_sigma_param_index(self):
        index = 2 * self.max_peaks + 2
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return 2 * self.max_peaks + 3
        else:
            return None


class PeriodicEMGModel(BaseModel):
    """
    Periodic Exponentially Modified Gaussian Model.

    Similar to the EMGModel, but with a single period parameter controlling all peak locations.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 3  # A, tau, w
        self.color = "orange"  # Unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        self.nDims = self.max_peaks * self.dim + 3  # A, tau, w, u0, T, sigma
        if self.fit_pulses:
            self.nDims += 1  # +1 for Npulse

        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$A_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$\tau_{{{}}}$".format(i + 1))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$w_{{{}}}$".format(i + 1))

        self.paramnames_all.append(r"$u_0$")
        self.paramnames_all.append(r"$T$")
        self.paramnames_all.append(r"$\sigma$")
        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse}}$")

    def model_function(self, t, theta, n):
        A = theta[0 : self.max_peaks]
        tau = theta[self.max_peaks : 2 * self.max_peaks]
        w = theta[2 * self.max_peaks : 3 * self.max_peaks]
        u0 = theta[3 * self.max_peaks]
        T = theta[3 * self.max_peaks + 1]

        u_n = u0 + n * T
        return (
            (A[n] / (2 * tau[n]))
            * np.exp(((u_n - t) / tau[n]) + ((2 * w[n] ** 2) / tau[n] ** 2))
            * erfc((((u_n - t) * tau[n]) + w[n] ** 2) / (w[n] * tau[n] * np.sqrt(2)))
        )

    @property
    def has_w(self):
        return True

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)

        uniform_prior = UniformPrior(0.001, 1.0)
        theta[: self.max_peaks] = uniform_prior(hypercube[: self.max_peaks])  # A_i
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )  # tau_i
        theta[2 * self.max_peaks : 3 * self.max_peaks] = uniform_prior(
            hypercube[2 * self.max_peaks : 3 * self.max_peaks]
        )  # w_i

        uniform_prior_u0 = UniformPrior(0.001, 4.0 / self.max_peaks)
        theta[3 * self.max_peaks] = uniform_prior_u0(
            hypercube[3 * self.max_peaks]
        )  # u0

        uniform_prior_T = UniformPrior(0.001, 4.0 / self.max_peaks)
        theta[3 * self.max_peaks + 1] = uniform_prior_T(
            hypercube[3 * self.max_peaks + 1]
        )  # T

        log_uniform_prior = LogUniformPrior(0.001, 1.0)
        sigma_index = 3 * self.max_peaks + 2
        theta[sigma_index] = log_uniform_prior(hypercube[sigma_index])  # sigma

        if self.fit_pulses:
            Npulse_index = sigma_index + 1
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            theta[Npulse_index] = int(Npulse_prior(hypercube[Npulse_index]))

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the periodic EMG model.
        """
        pp = data["pp"]
        t = data["t"]

        sigma = theta[3 * self.max_peaks + 2]
        if self.fit_pulses:
            Npulse = int(theta[3 * self.max_peaks + 3])
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        for n in range(Npulse):
            pp_ += self.model_function(t, theta, n)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def get_period_param_indices(self):
        return [3 * self.max_peaks + 1]

    def get_u0_param_indices(self):
        return [3 * self.max_peaks]

    def get_sigma_param_index(self):
        index = 3 * self.max_peaks + 2
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return 3 * self.max_peaks + 3
        else:
            return None


def get_model(model_name, settings):
    """
    Factory function to get the appropriate model instance based on the model name.
    """
    if model_name == "emg":
        return EMGModel(settings)
    elif model_name == "exponential":
        return ExponentialModel(settings)
    elif model_name == "periodic_exponential":
        return PeriodicExponentialModel(settings)
    elif model_name == "periodic_emg":
        return PeriodicEMGModel(settings)
    elif model_name == "double_periodic_exp":
        return DoublePeriodicExponentialModel(settings)
    elif model_name == "periodic_exp_plus_exp":
        return PeriodicExponentialPlusExponentialModel(settings)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
