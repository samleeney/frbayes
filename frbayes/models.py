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

    def get_Npulse_param_index(self):
        """
        Returns the index of Npulse parameter in theta.
        """
        return None


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
        self.paramnames_all = []  # Reset to avoid duplication
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

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)
        idx = 0  # Index tracker for hypercube

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("emg", "amplitude")
        tau_range = self.settings.get_prior_range("emg", "tau")
        u_range = self.settings.get_prior_range("emg", "u")
        width_range = self.settings.get_prior_range("emg", "width")
        sigma_range = self.settings.get_prior_range("emg", "sigma")

        # Sample amplitudes A_i
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.max_peaks] = uniform_prior_A(hypercube[idx : idx + self.max_peaks])
        idx += self.max_peaks

        # Sample decay times τ_i
        uniform_prior_tau = UniformPrior(tau_range["min"], tau_range["max"])
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior_tau(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample arrival times u_i
        u_hypercube = hypercube[idx : idx + self.max_peaks]
        idx += self.max_peaks

        # Sample width parameters w_i
        uniform_prior_w = UniformPrior(width_range["min"], width_range["max"])
        theta[3 * self.max_peaks : 4 * self.max_peaks] = uniform_prior_w(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample sigma
        log_uniform_prior_sigma = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[4 * self.max_peaks] = log_uniform_prior_sigma(hypercube[idx])
        idx += 1

        if self.fit_pulses:
            # Sample Npulse
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            Npulse = Npulse_prior(hypercube[idx])
            idx += 1
        else:
            Npulse = self.max_peaks

        # Assign u_i to theta using Npulse
        if int(Npulse) > 0:
            # Using sorted prior for active pulses
            sorted_prior_u = SortedUniformPrior(u_range["min"], u_range["max"])
            active_u = sorted_prior_u(u_hypercube[:int(Npulse)])
            theta[2 * self.max_peaks : 2 * self.max_peaks + int(Npulse)] = active_u

            if int(Npulse) < self.max_peaks:
                # Uniform prior for inactive pulses
                uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
                inactive_u = uniform_prior_u(u_hypercube[int(Npulse):])
                theta[2 * self.max_peaks + int(Npulse) : 3 * self.max_peaks] = inactive_u
        else:
            # If Npulse is 0 (unlikely), all u_i are uniform
            uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
            theta[2 * self.max_peaks : 3 * self.max_peaks] = uniform_prior_u(u_hypercube)

        if self.fit_pulses:
            theta[-1] = Npulse  # Npulse is the last parameter

        return theta

    def loglikelihood(self, theta, data):
        pp = data["pp"]
        t = data["t"]

        sigma = theta[4 * self.max_peaks]
        if self.fit_pulses:
            Npulse = theta[-1]  # Npulse is the last parameter
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for i in range(int(Npulse)):
            pp_ += self.model_function(t, theta, i)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff ** 2) / (sigma ** 2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

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

        exp_arg = ((u[i] - t) / tau[i]) + ((2 * w[i] ** 2) / (tau[i] ** 2))
        exp_val = np.exp(exp_arg)

        erfc_arg = ((((u[i] - t) * tau[i]) + w[i] ** 2) / (w[i] * tau[i] * np.sqrt(2)))
        erfc_val = erfc(erfc_arg)

        return (A[i] / (2 * tau[i])) * exp_val * erfc_val

    @property
    def has_w(self):
        return True

    def get_sigma_param_index(self):
        index = 4 * self.max_peaks
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return self.nDims - 1  # Npulse is the last parameter
        else:
            return None


class EMGModelWithBaseline(BaseModel):
    """
    Exponentially Modified Gaussian (EMG) Model with a constant baseline offset.

    Inherits from BaseModel and implements the EMG model specifics with an added baseline.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 4  # Number of parameters per peak for EMG model
        self.color = "cyan"  # Assign a unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        """
        Setup model parameters and names.
        """
        self.paramnames_all = []  # Reset to avoid duplication
        self.nDims = self.max_peaks * self.dim + 2  # +2 for sigma and baseline
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
        self.paramnames_all.append(r"$B_{\text{offset}}$") # Baseline offset
        self.paramnames_all.append(r"$\sigma$")
        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse}}$")

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)
        idx = 0  # Index tracker for hypercube

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("emg", "amplitude")
        tau_range = self.settings.get_prior_range("emg", "tau")
        u_range = self.settings.get_prior_range("emg", "u")
        width_range = self.settings.get_prior_range("emg", "width")
        baseline_range = self.settings.get_prior_range("emg_with_baseline", "baseline_offset")
        sigma_range = self.settings.get_prior_range("emg", "sigma")

        # Sample amplitudes A_i
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.max_peaks] = uniform_prior_A(hypercube[idx : idx + self.max_peaks])
        idx += self.max_peaks

        # Sample decay times τ_i
        uniform_prior_tau = UniformPrior(tau_range["min"], tau_range["max"])
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior_tau(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample arrival times u_i
        u_hypercube = hypercube[idx : idx + self.max_peaks]
        idx += self.max_peaks

        # Sample width parameters w_i
        uniform_prior_w = UniformPrior(width_range["min"], width_range["max"])
        theta[3 * self.max_peaks : 4 * self.max_peaks] = uniform_prior_w(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample baseline offset
        uniform_prior_baseline = UniformPrior(baseline_range["min"], baseline_range["max"])
        theta[4 * self.max_peaks] = uniform_prior_baseline(hypercube[idx])
        idx += 1

        # Sample sigma
        log_uniform_prior_sigma = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[4 * self.max_peaks + 1] = log_uniform_prior_sigma(hypercube[idx])
        idx += 1

        if self.fit_pulses:
            # Sample Npulse
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            Npulse = Npulse_prior(hypercube[idx])
            idx += 1
        else:
            Npulse = self.max_peaks

        # Assign u_i to theta using Npulse
        if int(Npulse) > 0:
            # Using sorted prior for active pulses
            sorted_prior_u = SortedUniformPrior(u_range["min"], u_range["max"])
            active_u = sorted_prior_u(u_hypercube[:int(Npulse)])
            theta[2 * self.max_peaks : 2 * self.max_peaks + int(Npulse)] = active_u

            if int(Npulse) < self.max_peaks:
                # Uniform prior for inactive pulses
                uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
                inactive_u = uniform_prior_u(u_hypercube[int(Npulse):])
                theta[2 * self.max_peaks + int(Npulse) : 3 * self.max_peaks] = inactive_u
        else:
            # If Npulse is 0 (unlikely), all u_i are uniform
            uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
            theta[2 * self.max_peaks : 3 * self.max_peaks] = uniform_prior_u(u_hypercube)

        if self.fit_pulses:
            theta[-1] = Npulse  # Npulse is the last parameter

        return theta

    def loglikelihood(self, theta, data):
        pp = data["pp"]
        t = data["t"]

        baseline_offset = theta[4 * self.max_peaks]
        sigma = theta[4 * self.max_peaks + 1]
        if self.fit_pulses:
            Npulse = theta[-1]  # Npulse is the last parameter
        else:
            Npulse = self.max_peaks

        pp_model_components = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for i in range(int(Npulse)):
            pp_model_components += self.model_function(t, theta, i)
        
        pp_ = pp_model_components + baseline_offset

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff ** 2) / (sigma ** 2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

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

        exp_arg = ((u[i] - t) / tau[i]) + ((2 * w[i] ** 2) / (tau[i] ** 2))
        exp_val = np.exp(exp_arg)

        erfc_arg = ((((u[i] - t) * tau[i]) + w[i] ** 2) / (w[i] * tau[i] * np.sqrt(2)))
        erfc_val = erfc(erfc_arg)

        return (A[i] / (2 * tau[i])) * exp_val * erfc_val

    @property
    def has_w(self):
        return True

    def get_baseline_param_index(self):
        return 4 * self.max_peaks

    def get_sigma_param_index(self):
        index = 4 * self.max_peaks + 1
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return self.nDims - 1  # Npulse is the last parameter
        else:
            return None


class EMGModel_wrong(BaseModel):
    """
    Exponentially Modified Gaussian (EMG) Model - WRONG version for testing.
    This version intentionally keeps an incorrect model_function.

    Inherits from BaseModel and implements the EMG model specifics.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 4  # Number of parameters per peak for EMG model
        self.color = "orange"  # Assign a unique color for plotting (distinct from correct EMG)
        self._setup_parameters()

    def _setup_parameters(self):
        """
        Setup model parameters and names.
        """
        self.paramnames_all = []  # Reset to avoid duplication
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

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)
        idx = 0  # Index tracker for hypercube

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("emg", "amplitude")
        tau_range = self.settings.get_prior_range("emg", "tau")
        u_range = self.settings.get_prior_range("emg", "u")
        width_range = self.settings.get_prior_range("emg", "width")
        sigma_range = self.settings.get_prior_range("emg", "sigma")

        # Sample amplitudes A_i
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.max_peaks] = uniform_prior_A(hypercube[idx : idx + self.max_peaks])
        idx += self.max_peaks

        # Sample decay times τ_i
        uniform_prior_tau = UniformPrior(tau_range["min"], tau_range["max"])
        theta[self.max_peaks : 2 * self.max_peaks] = uniform_prior_tau(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample arrival times u_i
        u_hypercube = hypercube[idx : idx + self.max_peaks]
        idx += self.max_peaks

        # Sample width parameters w_i
        uniform_prior_w = UniformPrior(width_range["min"], width_range["max"])
        theta[3 * self.max_peaks : 4 * self.max_peaks] = uniform_prior_w(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample sigma
        log_uniform_prior_sigma = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[4 * self.max_peaks] = log_uniform_prior_sigma(hypercube[idx])
        idx += 1

        if self.fit_pulses:
            # Sample Npulse
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            Npulse = Npulse_prior(hypercube[idx])
            idx += 1
        else:
            Npulse = self.max_peaks

        # Assign u_i to theta using Npulse
        if int(Npulse) > 0:
            # Using sorted prior for active pulses
            sorted_prior_u = SortedUniformPrior(u_range["min"], u_range["max"])
            active_u = sorted_prior_u(u_hypercube[:int(Npulse)])
            theta[2 * self.max_peaks : 2 * self.max_peaks + int(Npulse)] = active_u

            if int(Npulse) < self.max_peaks:
                # Uniform prior for inactive pulses
                uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
                inactive_u = uniform_prior_u(u_hypercube[int(Npulse):])
                theta[2 * self.max_peaks + int(Npulse) : 3 * self.max_peaks] = inactive_u
        else:
            # If Npulse is 0 (unlikely), all u_i are uniform
            uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
            theta[2 * self.max_peaks : 3 * self.max_peaks] = uniform_prior_u(u_hypercube)

        if self.fit_pulses:
            theta[-1] = Npulse  # Npulse is the last parameter

        return theta

    def loglikelihood(self, theta, data):
        pp = data["pp"]
        t = data["t"]

        sigma = theta[4 * self.max_peaks]
        if self.fit_pulses:
            Npulse = theta[-1]  # Npulse is the last parameter
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for i in range(int(Npulse)):
            pp_ += self.model_function(t, theta, i)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff ** 2) / (sigma ** 2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def model_function(self, t, theta, i):
        """
        Compute the EMG model function for peak i (INCORRECT VERSION).

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
            * np.exp(((u[i] - t) / tau[i]) + ((w[i] ** 2) / (2 * tau[i] ** 2)))  # Original "wrong" term
            * erfc((((u[i] - t) * tau[i]) + w[i] ** 2) / (w[i] * tau[i] * np.sqrt(2)))
        )

    @property
    def has_w(self):
        return True

    def get_sigma_param_index(self):
        index = 4 * self.max_peaks
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return self.nDims - 1  # Npulse is the last parameter
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
        self.paramnames_all = []  # Reset to avoid duplication
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

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)
        idx = 0  # Index tracker for hypercube

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("exponential", "amplitude")
        tau_range = self.settings.get_prior_range("exponential", "tau")
        u_range = self.settings.get_prior_range("exponential", "u")
        sigma_range = self.settings.get_prior_range("exponential", "sigma")

        # Sample amplitudes A_i
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.max_peaks] = uniform_prior_A(hypercube[idx : idx + self.max_peaks])
        idx += self.max_peaks

        # Sample decay times τ_i
        log_uniform_prior_tau = LogUniformPrior(tau_range["min"], tau_range["max"])
        theta[self.max_peaks : 2 * self.max_peaks] = log_uniform_prior_tau(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample arrival times u_i
        u_hypercube = hypercube[idx : idx + self.max_peaks]
        idx += self.max_peaks

        # Sample sigma
        log_uniform_prior_sigma = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[3 * self.max_peaks] = log_uniform_prior_sigma(hypercube[idx])
        idx += 1

        if self.fit_pulses:
            # Sample Npulse
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            Npulse = Npulse_prior(hypercube[idx])
            idx += 1
        else:
            Npulse = self.max_peaks

        # Assign u_i to theta using Npulse
        if int(Npulse) > 0:
            # Using sorted prior for active pulses
            sorted_prior_u = SortedUniformPrior(u_range["min"], u_range["max"])
            active_u = sorted_prior_u(u_hypercube[:int(Npulse)])
            theta[2 * self.max_peaks : 2 * self.max_peaks + int(Npulse)] = active_u

            if int(Npulse) < self.max_peaks:
                # Uniform prior for inactive pulses
                uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
                inactive_u = uniform_prior_u(u_hypercube[int(Npulse):])
                theta[2 * self.max_peaks + int(Npulse) : 3 * self.max_peaks] = inactive_u
        else:
            # If Npulse is 0 (unlikely), all u_i are uniform
            uniform_prior_u = UniformPrior(u_range["min"], u_range["max"])
            theta[2 * self.max_peaks : 3 * self.max_peaks] = uniform_prior_u(u_hypercube)

        if self.fit_pulses:
            theta[-1] = Npulse  # Npulse is the last parameter

        return theta

    def loglikelihood(self, theta, data):
        pp = data["pp"]
        t = data["t"]

        sigma = theta[3 * self.max_peaks]
        if self.fit_pulses:
            Npulse = theta[-1]  # Npulse is the last parameter
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for i in range(int(Npulse)):
            pp_ += self.model_function(t, theta, i)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff ** 2) / (sigma ** 2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def model_function(self, t, theta, i):
        """
        Compute the exponential model function for peak i.
        """
        A = theta[0 : self.max_peaks]
        tau = theta[self.max_peaks : 2 * self.max_peaks]
        u = theta[2 * self.max_peaks : 3 * self.max_peaks]

        return np.where(t <= u[i], 0.0, A[i] * np.exp(-(t - u[i]) / tau[i]))

    def get_sigma_param_index(self):
        index = 3 * self.max_peaks
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            return self.nDims - 1  # Npulse is the last parameter
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
        self.color = "green"  # Assign a unique color for plotting
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

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("periodic_exponential", "amplitude")
        tau_range = self.settings.get_prior_range("periodic_exponential", "tau")
        sigma_range = self.settings.get_prior_range("periodic_exponential", "sigma")
        
        # Get periodic-specific ranges
        periodic_range = self.settings.get_prior_range("periodic", "period")
        u0_range = self.settings.get_prior_range("periodic", "u0")
        
        # Calculate max values for u0 and period if not specified
        u0_max = u0_range["max"] if u0_range["max"] is not None else 4.0 / self.max_peaks
        period_max = periodic_range["max"] if periodic_range["max"] is not None else 4.0 / self.max_peaks

        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.max_peaks] = uniform_prior_A(hypercube[: self.max_peaks])  # A_i
        
        log_uniform_prior_tau = LogUniformPrior(tau_range["min"], tau_range["max"])
        theta[self.max_peaks : 2 * self.max_peaks] = log_uniform_prior_tau(
            hypercube[self.max_peaks : 2 * self.max_peaks]
        )  # tau_i

        uniform_prior_u0 = UniformPrior(u0_range["min"], u0_max)
        theta[2 * self.max_peaks] = uniform_prior_u0(
            hypercube[2 * self.max_peaks]
        )  # u0

        uniform_prior_T = UniformPrior(periodic_range["min"], period_max)
        theta[2 * self.max_peaks + 1] = uniform_prior_T(
            hypercube[2 * self.max_peaks + 1]
        )  # T

        log_uniform_prior = LogUniformPrior(sigma_range["min"], sigma_range["max"])
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
            Npulse = theta[2 * self.max_peaks + 3]
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for n in range(int(Npulse)):
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
            return self.nDims - 1  # Npulse is the last parameter
        else:
            return None


class PeriodicExponentialPlusExponentialModel(BaseModel):
    """
    Combination of Periodic Exponential Model and Exponential Model.

    This model combines a PeriodicExponentialModel with an ExponentialModel.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim_periodic = 2  # A and tau per peak for periodic model
        self.dim_exponential = 3  # A, tau, u per peak for exponential model
        self.color = "brown"  # Unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        self.max_peaks_total = self.max_peaks
        self.n1 = self.max_peaks_total // 2  # Number of peaks in periodic model
        self.n2 = self.max_peaks_total - self.n1  # Number of peaks in exponential model

        self.nDims = (
            self.n1 * self.dim_periodic
            + self.n2 * self.dim_exponential
            + 3  # u0, T, sigma
        )
        if self.fit_pulses:
            self.nDims += 2  # Npulse_periodic, Npulse_exponential

        # PeriodicExponentialModel parameters
        for i in range(self.n1):
            self.paramnames_all.append(r"$A_{\text{per},{" + str(i + 1) + r"}}$")
        for i in range(self.n1):
            self.paramnames_all.append(r"$\tau_{\text{per},{" + str(i + 1) + r"}}$")

        self.paramnames_all.append(r"$u_0$")
        self.paramnames_all.append(r"$T$")

        # ExponentialModel parameters
        for i in range(self.n2):
            self.paramnames_all.append(r"$A_{\text{exp},{" + str(i + 1) + r"}}$")
        for i in range(self.n2):
            self.paramnames_all.append(r"$\tau_{\text{exp},{" + str(i + 1) + r"}}$")
        for i in range(self.n2):
            self.paramnames_all.append(r"$u_{\text{exp},{" + str(i + 1) + r"}}$")

        self.paramnames_all.append(r"$\sigma$")
        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse,per}}$")
            self.paramnames_all.append(r"$N_{\text{pulse,exp}}$")

    def model_function(self, t, theta):
        # Periodic part
        A_per = theta[0 : self.n1]
        tau_per = theta[self.n1 : 2 * self.n1]
        u0 = theta[2 * self.n1]
        T = theta[2 * self.n1 + 1]

        # Exponential part
        start_exp_params = 2 * self.n1 + 2
        A_exp = theta[start_exp_params : start_exp_params + self.n2]
        tau_exp = theta[start_exp_params + self.n2 : start_exp_params + 2 * self.n2]
        u_exp = theta[start_exp_params + 2 * self.n2 : start_exp_params + 3 * self.n2]

        # Sigma and Npulse indices
        sigma_index = start_exp_params + 3 * self.n2
        sigma = theta[sigma_index]

        if self.fit_pulses:
            Npulse_per_index = sigma_index + 1
            Npulse_exp_index = sigma_index + 2
            Npulse_per = theta[Npulse_per_index]
            Npulse_exp = theta[Npulse_exp_index]
        else:
            Npulse_per = self.n1
            Npulse_exp = self.n2

        f_per = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for n in range(int(Npulse_per)):
            u_n = u0 + n * T
            f = A_per[n] * np.exp(-(t - u_n) / tau_per[n])
            f = np.where(t <= u_n, 0.0, f)
            f_per += f

        f_exp = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for i in range(int(Npulse_exp)):
            f = A_exp[i] * np.exp(-(t - u_exp[i]) / tau_exp[i])
            f = np.where(t <= u_exp[i], 0.0, f)
            f_exp += f

        return f_per + f_exp, f_per, f_exp  # Return individual components as well

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("periodic_exp_plus_exp", "amplitude")
        tau_range = self.settings.get_prior_range("periodic_exp_plus_exp", "tau")
        u_range = self.settings.get_prior_range("periodic_exp_plus_exp", "u")
        sigma_range = self.settings.get_prior_range("periodic_exp_plus_exp", "sigma")
        
        # Get periodic-specific ranges
        periodic_range = self.settings.get_prior_range("periodic", "period")
        u0_range = self.settings.get_prior_range("periodic", "u0")
        
        # Calculate max values for u0 and period if not specified
        u0_max = u0_range["max"] if u0_range["max"] is not None else 4.0 / self.n1
        period_max = periodic_range["max"] if periodic_range["max"] is not None else 4.0 / self.n1

        # Periodic model parameters
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.n1] = uniform_prior_A(hypercube[: self.n1])  # A_per_i
        
        log_uniform_prior_tau = LogUniformPrior(tau_range["min"], tau_range["max"])
        theta[self.n1 : 2 * self.n1] = log_uniform_prior_tau(
            hypercube[self.n1 : 2 * self.n1]
        )  # tau_per_i

        uniform_prior_u0 = UniformPrior(u0_range["min"], u0_max)
        theta[2 * self.n1] = uniform_prior_u0(hypercube[2 * self.n1])  # u0

        uniform_prior_T = UniformPrior(periodic_range["min"], period_max)
        theta[2 * self.n1 + 1] = uniform_prior_T(hypercube[2 * self.n1 + 1])  # T

        # Exponential model parameters
        start_exp_hypercube = 2 * self.n1 + 2
        start_exp_theta = 2 * self.n1 + 2

        theta[start_exp_theta : start_exp_theta + self.n2] = uniform_prior_A(
            hypercube[start_exp_hypercube : start_exp_hypercube + self.n2]
        )  # A_exp_i
        theta[start_exp_theta + self.n2 : start_exp_theta + 2 * self.n2] = (
            log_uniform_prior_tau(
                hypercube[
                    start_exp_hypercube
                    + self.n2 : start_exp_hypercube
                    + 2 * self.n2
                ]
            )
        )  # tau_exp_i
        sorted_uniform_prior = SortedUniformPrior(u_range["min"], u_range["max"])
        theta[start_exp_theta + 2 * self.n2 : start_exp_theta + 3 * self.n2] = (
            sorted_uniform_prior(
                hypercube[
                    start_exp_hypercube
                    + 2 * self.n2 : start_exp_hypercube
                    + 3 * self.n2
                ]
            )
        )  # u_exp_i

        # Sigma
        sigma_index = start_exp_theta + 3 * self.n2
        log_uniform_prior = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[sigma_index] = log_uniform_prior(hypercube[sigma_index])  # sigma

        if self.fit_pulses:
            Npulse_per_index = sigma_index + 1
            Npulse_exp_index = sigma_index + 2
            Npulse_prior_per = UniformPrior(1, self.n1 + 1)
            Npulse_prior_exp = UniformPrior(1, self.n2 + 1)
            theta[Npulse_per_index] = Npulse_prior_per(hypercube[Npulse_per_index])
            theta[Npulse_exp_index] = Npulse_prior_exp(hypercube[Npulse_exp_index])

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the combined model.
        """
        pp = data["pp"]
        t = data["t"]

        pp_, f_per, f_exp = self.model_function(t, theta)
        diff = pp - pp_

        sigma_index = (
            2 * self.n1 + 2 + 3 * self.n2
        )  # After periodic and exponential params
        sigma = theta[sigma_index]

        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def get_period_param_indices(self):
        return [2 * self.n1 + 1]

    def get_u0_param_indices(self):
        return [2 * self.n1]

    def get_sigma_param_index(self):
        index = 2 * self.n1 + 2 + 3 * self.n2
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            index = self.get_sigma_param_index()
            return [index + 1, index + 2]
        else:
            return None


class DoublePeriodicExponentialModel(BaseModel):
    """
    Combination of two Periodic Exponential Models.

    This model combines two PeriodicExponentialModels.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 2  # A and tau per peak
        self.color = "magenta"  # Unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        self.max_peaks_total = self.max_peaks
        self.n1 = self.max_peaks_total // 2  # Number of peaks in first periodic model
        self.n2 = (
            self.max_peaks_total - self.n1
        )  # Number of peaks in second periodic model

        self.nDims = (
            self.n1 * self.dim
            + self.n2 * self.dim
            + 4  # u0_1, T1, u0_2, T2
            + 1  # sigma
        )
        if self.fit_pulses:
            self.nDims += 2  # Npulse_1, Npulse_2

        # First PeriodicExponentialModel parameters
        for i in range(self.n1):
            self.paramnames_all.append(r"$A_{1,{" + str(i + 1) + r"}}$")
        for i in range(self.n1):
            self.paramnames_all.append(r"$\tau_{1,{" + str(i + 1) + r"}}$")
        self.paramnames_all.append(r"$u_{0,1}$")
        self.paramnames_all.append(r"$T_{1}$")

        # Second PeriodicExponentialModel parameters
        for i in range(self.n2):
            self.paramnames_all.append(r"$A_{2,{" + str(i + 1) + r"}}$")
        for i in range(self.n2):
            self.paramnames_all.append(r"$\tau_{2,{" + str(i + 1) + r"}}$")
        self.paramnames_all.append(r"$u_{0,2}$")
        self.paramnames_all.append(r"$T_{2}$")

        self.paramnames_all.append(r"$\sigma$")
        if self.fit_pulses:
            self.paramnames_all.append(r"$N_{\text{pulse,1}}$")
            self.paramnames_all.append(r"$N_{\text{pulse,2}}$")

    def model_function(self, t, theta):
        # First periodic model
        A1 = theta[0 : self.n1]
        tau1 = theta[self.n1 : 2 * self.n1]
        u0_1 = theta[2 * self.n1]
        T1 = theta[2 * self.n1 + 1]

        # Second periodic model
        start_second_model = 2 * self.n1 + 2
        A2 = theta[start_second_model : start_second_model + self.n2]
        tau2 = theta[start_second_model + self.n2 : start_second_model + 2 * self.n2]
        u0_2 = theta[start_second_model + 2 * self.n2]
        T2 = theta[start_second_model + 2 * self.n2 + 1]

        # Sigma and Npulse indices
        sigma_index = start_second_model + 2 * self.n2 + 2
        sigma = theta[sigma_index]

        if self.fit_pulses:
            Npulse_1_index = sigma_index + 1
            Npulse_2_index = sigma_index + 2
            Npulse_1 = theta[Npulse_1_index]
            Npulse_2 = theta[Npulse_2_index]
        else:
            Npulse_1 = self.n1
            Npulse_2 = self.n2

        f1 = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for n in range(int(Npulse_1)):
            u_n = u0_1 + n * T1
            f = A1[n] * np.exp(-(t - u_n) / tau1[n])
            f = np.where(t <= u_n, 0.0, f)
            f1 += f

        f2 = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for n in range(int(Npulse_2)):
            u_n = u0_2 + n * T2
            f = A2[n] * np.exp(-(t - u_n) / tau2[n])
            f = np.where(t <= u_n, 0.0, f)
            f2 += f

        return f1 + f2, f1, f2  # Return individual components as well

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("double_periodic_exp", "amplitude")
        tau_range = self.settings.get_prior_range("double_periodic_exp", "tau")
        sigma_range = self.settings.get_prior_range("double_periodic_exp", "sigma")
        
        # Get periodic-specific ranges
        periodic_range = self.settings.get_prior_range("periodic", "period")
        u0_range = self.settings.get_prior_range("periodic", "u0")
        
        # Calculate max values for u0 and period if not specified
        u0_max_1 = u0_range["max"] if u0_range["max"] is not None else 4.0 / self.n1
        period_max_1 = periodic_range["max"] if periodic_range["max"] is not None else 4.0 / self.n1
        u0_max_2 = u0_range["max"] if u0_range["max"] is not None else 4.0 / self.n2
        period_max_2 = periodic_range["max"] if periodic_range["max"] is not None else 4.0 / self.n2

        # First periodic model parameters
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.n1] = uniform_prior_A(hypercube[: self.n1])  # A1_i
        
        log_uniform_prior_tau = LogUniformPrior(tau_range["min"], tau_range["max"])
        theta[self.n1 : 2 * self.n1] = log_uniform_prior_tau(
            hypercube[self.n1 : 2 * self.n1]
        )  # tau1_i

        uniform_prior_u0 = UniformPrior(u0_range["min"], u0_max_1)
        theta[2 * self.n1] = uniform_prior_u0(hypercube[2 * self.n1])  # u0_1

        uniform_prior_T = UniformPrior(periodic_range["min"], period_max_1)
        theta[2 * self.n1 + 1] = uniform_prior_T(hypercube[2 * self.n1 + 1])  # T1

        # Second periodic model parameters
        start_second_hypercube = 2 * self.n1 + 2
        start_second_theta = 2 * self.n1 + 2

        theta[start_second_theta : start_second_theta + self.n2] = uniform_prior_A(
            hypercube[start_second_hypercube : start_second_hypercube + self.n2]
        )  # A2_i
        theta[start_second_theta + self.n2 : start_second_theta + 2 * self.n2] = (
            log_uniform_prior_tau(
                hypercube[
                    start_second_hypercube
                    + self.n2 : start_second_hypercube
                    + 2 * self.n2
                ]
            )
        )  # tau2_i

        uniform_prior_u0_2 = UniformPrior(u0_range["min"], u0_max_2)
        theta[start_second_theta + 2 * self.n2] = uniform_prior_u0_2(
            hypercube[start_second_hypercube + 2 * self.n2]
        )  # u0_2

        uniform_prior_T2 = UniformPrior(periodic_range["min"], period_max_2)
        theta[start_second_theta + 2 * self.n2 + 1] = uniform_prior_T2(
            hypercube[start_second_hypercube + 2 * self.n2 + 1]
        )  # T2

        # Sigma
        sigma_index = start_second_theta + 2 * self.n2 + 2
        log_uniform_prior = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[sigma_index] = log_uniform_prior(hypercube[sigma_index])  # sigma

        if self.fit_pulses:
            Npulse_1_index = sigma_index + 1
            Npulse_2_index = sigma_index + 2
            Npulse_prior_1 = UniformPrior(1, self.n1 + 1)
            Npulse_prior_2 = UniformPrior(1, self.n2 + 1)
            theta[Npulse_1_index] = Npulse_prior_1(hypercube[Npulse_1_index])
            theta[Npulse_2_index] = Npulse_prior_2(hypercube[Npulse_2_index])

        return theta

    def loglikelihood(self, theta, data):
        """
        Compute the log-likelihood for the combined model.
        """
        pp = data["pp"]
        t = data["t"]

        pp_, f1, f2 = self.model_function(t, theta)
        diff = pp - pp_

        sigma_index = (
            2 * self.n1 + 2 + 2 * self.n2 + 2
        )  # After first and second periodic params
        sigma = theta[sigma_index]

        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    def get_period_param_indices(self):
        index_T1 = 2 * self.n1 + 1
        index_T2 = 2 * self.n1 + 2 + 2 * self.n2 + 1
        return [index_T1, index_T2]

    def get_u0_param_indices(self):
        index_u0_1 = 2 * self.n1
        index_u0_2 = 2 * self.n1 + 2 + 2 * self.n2
        return [index_u0_1, index_u0_2]

    def get_sigma_param_index(self):
        index = 2 * self.n1 + 2 + 2 * self.n2 + 2
        return index

    def get_Npulse_param_index(self):
        if self.fit_pulses:
            index = self.get_sigma_param_index()
            return [index + 1, index + 2]
        else:
            return None


class PeriodicEMGModel(BaseModel):
    """
    Periodic Exponentially Modified Gaussian (EMG) Model.

    Similar to the EMGModel, but with a single period parameter controlling all peak locations.
    """

    def __init__(self, settings):
        super().__init__(settings)
        self.dim = 3  # A, tau, and w per peak
        self.color = "purple"  # Assign a unique color for plotting
        self._setup_parameters()

    def _setup_parameters(self):
        self.nDims = self.max_peaks * self.dim + 3  # A*npeak, tau*npeak, w*npeak, u0, T, sigma
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

    def prior(self, hypercube):
        theta = np.zeros(self.nDims)
        idx = 0  # Index tracker for hypercube

        # Get prior ranges from settings
        amplitude_range = self.settings.get_prior_range("periodic_emg", "amplitude")
        tau_range = self.settings.get_prior_range("periodic_emg", "tau")
        width_range = self.settings.get_prior_range("periodic_emg", "width")
        sigma_range = self.settings.get_prior_range("periodic_emg", "sigma")
        
        # Get periodic-specific ranges
        periodic_range = self.settings.get_prior_range("periodic", "period")
        u0_range = self.settings.get_prior_range("periodic", "u0")
        
        # Calculate max values for u0 and period if not specified
        u0_max = u0_range["max"] if u0_range["max"] is not None else 4.0 / self.max_peaks
        period_max = periodic_range["max"] if periodic_range["max"] is not None else 4.0 / self.max_peaks

        # Sample amplitudes A_i
        uniform_prior_A = UniformPrior(amplitude_range["min"], amplitude_range["max"])
        theta[: self.max_peaks] = uniform_prior_A(hypercube[idx : idx + self.max_peaks])
        idx += self.max_peaks

        # Sample decay times τ_i
        log_uniform_prior_tau = LogUniformPrior(tau_range["min"], tau_range["max"])
        theta[self.max_peaks : 2 * self.max_peaks] = log_uniform_prior_tau(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample width parameters w_i
        uniform_prior_w = UniformPrior(width_range["min"], width_range["max"])
        theta[2 * self.max_peaks : 3 * self.max_peaks] = uniform_prior_w(
            hypercube[idx : idx + self.max_peaks]
        )
        idx += self.max_peaks

        # Sample u0
        uniform_prior_u0 = UniformPrior(u0_range["min"], u0_max)
        theta[3 * self.max_peaks] = uniform_prior_u0(hypercube[idx])
        idx += 1

        # Sample T
        uniform_prior_T = UniformPrior(periodic_range["min"], period_max)
        theta[3 * self.max_peaks + 1] = uniform_prior_T(hypercube[idx])
        idx += 1

        # Sample sigma
        log_uniform_prior_sigma = LogUniformPrior(sigma_range["min"], sigma_range["max"])
        theta[3 * self.max_peaks + 2] = log_uniform_prior_sigma(hypercube[idx])
        idx += 1

        if self.fit_pulses:
            # Sample Npulse
            Npulse_prior = UniformPrior(1, self.max_peaks + 1)
            theta[3 * self.max_peaks + 3] = Npulse_prior(hypercube[idx])

        return theta

    def loglikelihood(self, theta, data):
        pp = data["pp"]
        t = data["t"]

        sigma = theta[3 * self.max_peaks + 2]
        if self.fit_pulses:
            Npulse = theta[3 * self.max_peaks + 3]
        else:
            Npulse = self.max_peaks

        pp_ = np.zeros(len(t))
        # Due to the size of the time array, it is not efficient to vectorize this loop
        for n in range(int(Npulse)):
            pp_ += self.model_function(t, theta, n)

        diff = pp - pp_
        logL = (-0.5 * np.sum((diff**2) / (sigma**2))) - (
            len(t) * np.log(sigma * np.sqrt(2 * np.pi))
        )

        return logL, []

    @property
    def has_w(self):
        return True

    def get_period_param_indices(self):
        return [3 * self.max_peaks + 1]

    def get_u0_param_indices(self):
        return [3 * self.max_peaks]

    def get_sigma_param_index(self):
        return 3 * self.max_peaks + 2

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
    elif model_name == "emg_with_baseline":
        return EMGModelWithBaseline(settings)
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
    elif model_name == "emg_wrong":
        return EMGModel_wrong(settings)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
