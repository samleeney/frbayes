import numpy as np
import pypolychord
from frbayes.analysis import FRBAnalysis
from frbayes.models import get_model
from frbayes.settings import global_settings

try:
    from mpi4py import MPI
except ImportError:
    pass


class FRBModel:
    """
    Class responsible for setting up and running the sampling process.

    Attributes:
        analysis (FRBAnalysis): Instance of FRBAnalysis containing data.
        pp (array): Pulse profile.
        t (array): Time axis.
        max_peaks (int): Maximum number of peaks.
        nDims (int): Number of dimensions/parameters in the model.
        model (BaseModel): Instance of the model used for sampling.
        fit_pulses (bool): Whether to fit the number of pulses.
    """

    def __init__(self):
        self.analysis = FRBAnalysis()
        self.pp = self.analysis.pulse_profile_snr
        self.t = self.analysis.time_axis
        self.max_peaks = global_settings.get("max_peaks")
        self.fit_pulses = global_settings.get("fit_pulses")

        # Get model instance
        model_name = global_settings.get("model")
        self.model = get_model(model_name, global_settings)
        self.nDims = self.model.nDims

    def loglikelihood(self, theta):
        """
        Compute the log-likelihood using the model's log-likelihood method.

        Args:
            theta (array): Model parameters.

        Returns:
            tuple: (log-likelihood value, empty list)
        """
        data = {"pp": self.pp, "t": self.t}
        return self.model.loglikelihood(theta, data)

    def prior(self, hypercube):
        """
        Transform unit hypercube samples to the parameter space using the model's prior.

        Args:
            hypercube (array): Samples from the unit hypercube.

        Returns:
            array: Transformed samples in the parameter space.
        """
        return self.model.prior(hypercube)

    def run_polychord(self):
        """
        Run the PolyChord nested sampling algorithm.
        """
        nDerived = 0

        output = pypolychord.run(
            self.loglikelihood,
            self.nDims,
            nDerived=nDerived,
            prior=self.prior,
            file_root=global_settings.get("file_root"),
            do_clustering=True,
            read_resume=True,
        )


if __name__ == "__main__":
    frb_model = FRBModel()
    frb_model.run_polychord()
