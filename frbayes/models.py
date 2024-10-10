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
        # Total dimensions
        self.nDims = self.max_peaks * self.dim + 1  # +1 for sigma
        if self.fit_pulses:
            self.nDims += 1  # +1 for Npulse

        # Define LaTeX-formatted parameter names
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$A_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$\tau_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$u_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$w_{{{}}}$".format(i))

        self.paramnames_all.append(r"$\sigma$")  # Noise parameter

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

    def prior(self, hypercube):
        """
        Transform unit hypercube samples to the parameter space for the EMG model.

        Args:
            hypercube (array): Samples from the unit hypercube.

        Returns:
            array: Transformed samples in the parameter space.
        """
        theta = np.zeros(self.nDims)

        # Amplitude A
        uniform_prior = UniformPrior(0.001, 5)
        theta[: self.max_peaks] = uniform_prior(hypercube[: self.max_peaks])

        # Time constant tau
        uniform_prior = UniformPrior(0.001, 5)
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )

        # Location u (sorted)
        sorted_uniform_prior = SortedUniformPrior(0.001, 3.9)
        theta[2 * self.max_peaks : 3 * self.max_peaks] = sorted_uniform_prior(
            hypercube[2 * self.max_peaks : 3 * self.max_peaks]
        )

        # Width w
        uniform_prior = UniformPrior(0.001, 5)
        theta[3 * self.max_peaks : 4 * self.max_peaks] = uniform_prior(
            hypercube[3 * self.max_peaks : 4 * self.max_peaks]
        )

        # Noise sigma
        log_uniform_prior = LogUniformPrior(0.001, 1)
        theta[4 * self.max_peaks] = log_uniform_prior(hypercube[4 * self.max_peaks])

        # Number of pulses Npulse
        if self.fit_pulses:
            theta[4 * self.max_peaks + 1] = UniformPrior(1, self.max_peaks)(
                hypercube[4 * self.max_peaks + 1]
            )

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the EMG model.

        Args:
            theta (array): Model parameters.
            data (dict): Dictionary containing 'pp' (pulse profile) and 't' (time axis).

        Returns:
            tuple: (log-likelihood value, empty list)
        """
        pp = data["pp"]
        t = data["t"]

        # Extract sigma
        sigma = theta[4 * self.max_peaks]

        # Determine number of pulses
        if self.fit_pulses:
            Npulse = int(theta[4 * self.max_peaks + 1])
        else:
            Npulse = self.max_peaks

        # Compute model prediction
        pp_ = np.zeros(len(t))
        for i in range(self.max_peaks):
            if i < Npulse:
                pp_ += self.model_function(t, theta, i)

        # Compute log-likelihood
        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []


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
        # Total dimensions
        self.nDims = self.max_peaks * self.dim + 1  # +1 for sigma
        if self.fit_pulses:
            self.nDims += 1  # +1 for Npulse

        # Define LaTeX-formatted parameter names
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$A_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$\tau_{{{}}}$".format(i))
        for i in range(self.max_peaks):
            self.paramnames_all.append(r"$u_{{{}}}$".format(i))

        self.paramnames_all.append(r"$\sigma$")  # Noise parameter

        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse}}$")

    def model_function(self, t, theta, i):
        """
        Compute the exponential model function for peak i.

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
        f = A[i] * np.exp(-(t - u[i]) / tau[i])
        f = np.where(t <= u[i], 0, f)
        return f

    def prior(self, hypercube):
        """
        Transform unit hypercube samples to the parameter space for the exponential model.

        Args:
            hypercube (array): Samples from the unit hypercube.

        Returns:
            array: Transformed samples in the parameter space.
        """
        theta = np.zeros(self.nDims)

        # Amplitude A
        uniform_prior = UniformPrior(0.001, 5)
        theta[: self.max_peaks] = uniform_prior(hypercube[: self.max_peaks])

        # Time constant tau
        uniform_prior = UniformPrior(0.001, 5)
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )

        # Location u (sorted)
        sorted_uniform_prior = SortedUniformPrior(0.001, 3.9)
        theta[2 * self.max_peaks : 3 * self.max_peaks] = sorted_uniform_prior(
            hypercube[2 * self.max_peaks : 3 * self.max_peaks]
        )

        # Noise sigma
        log_uniform_prior = LogUniformPrior(0.001, 1)
        theta[3 * self.max_peaks] = log_uniform_prior(hypercube[3 * self.max_peaks])

        # Number of pulses Npulse
        if self.fit_pulses:
            theta[3 * self.max_peaks + 1] = UniformPrior(1, self.max_peaks)(
                hypercube[3 * self.max_peaks + 1]
            )

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the exponential model.

        Args:
            theta (array): Model parameters.
            data (dict): Dictionary containing 'pp' (pulse profile) and 't' (time axis).

        Returns:
            tuple: (log-likelihood value, empty list)
        """
        pp = data["pp"]
        t = data["t"]

        # Extract sigma
        sigma = theta[3 * self.max_peaks]

        # Determine number of pulses
        if self.fit_pulses:
            Npulse = int(theta[3 * self.max_peaks + 1])
        else:
            Npulse = self.max_peaks

        # Compute model prediction
        pp_ = np.zeros(len(t))
        for i in range(self.max_peaks):
            if i < Npulse:
                pp_ += self.model_function(t, theta, i)

        # Compute log-likelihood
        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []


def get_model(model_name, settings):
    """
    Factory function to get the appropriate model instance based on the model name.

    Args:
        model_name (str): Name of the model.
        settings (Settings): Global settings instance.

    Returns:
        BaseModel: An instance of a subclass of BaseModel corresponding to the model name.
    """
    if model_name == "emg":
        return EMGModel(settings)
    elif model_name == "exponential":
        return ExponentialModel(settings)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
