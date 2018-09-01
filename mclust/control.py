import numpy as np

from mclust.models import Model


class EMControl:
    """
    Settings object for ME models
    """
    def __init__(self, eps=None, tol=None, itmax=None, equalPro=False):
        """
        Constructor

        :param eps: A scalar tolerance associated with deciding when to terminate
                    computations due to computational singularity in covariances.
                    Smaller values of eps allow computations to proceed nearer to
                    singularity, the default value is the relative machine precision.
        :param tol: A vector of length two giving relative convergence tolerances for
                    the log-likelihood and for parameter convergence in the inner loop
                    for models with iterative M-step ("VEI", "EVE", "VEE", "VVE", "VEV"),
                    respectively.
        :param itmax: A vector of length two giving integer limits on the number of EM
                      iterations and on the number of iterations in the inner loop for
                      models with iterative M-step ("VEI", "EVE", "VEE", "VVE", "VEV"),
                      respectively.
        :param equalPro: Logical variable indicating whether or not the mixing proportions are equal in the model.
        """
        if eps is None:
            self.eps = np.finfo(float).eps
        else:
            if not 0 <= eps < 1:
                raise ArithmeticError("eps should be between in (0, 1]")
            self.eps = eps
        if tol is None:
            self.tol = [1.0e-05, np.sqrt(np.finfo(float).eps)]
        else:
            if any(not (0 <= toli < 1) for toli in tol):
                raise ArithmeticError("tol should be between in (0, 1]")
            if len(tol) == 1:
                tol = tol + tol
            self.tol = tol
        if itmax is None:
            self.itmax = [2147483647, 2147483647]
        else:
            if any(it < 0 for it in itmax):
                raise ArithmeticError("itmax is negative")
            if len(itmax) == 1:
                itmax = [itmax, 2147483647]
            self.itmax = itmax
        self.equalPro = equalPro


class ModelTypes:
    """
    Helper class to select often occuring groups of models
    """
    @staticmethod
    def get_one_dimensional():
        """
        One dimensional models.
        :return:  List with all one dimensional model configurations.
        """
        return [Model.E, Model.V]

    @staticmethod
    def get_multi_dimensional():
        """
        All multidimensional models.
        :return: List with all multidimensional model configurations.
        """
        return [Model.EII, Model.VII, Model.EEI, Model.VEI,
                Model.EVI, Model.VVI, Model.EEE, Model.EVE,
                Model.VEE, Model.VVE, Model.EEV, Model.VEV,
                Model.EVV, Model.VVV]

    @staticmethod
    def get_less_observations():
        """
        Subset multidimensional models for data with less observations than dimensions.
        :return: List of multidimensional model configurations for data with n < d.
        """
        return [Model.EEE, Model.EEV, Model.VEV, Model.VVV]

