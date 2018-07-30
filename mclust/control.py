import numpy as np
from math import sqrt

from mclust.models import Model


class EMControl:
    def __init__(self, eps=None, tol=None, itmax=None, equalPro=False):
        if eps is None:
            self.eps = np.finfo(float).eps
        else:
            if not 0 <= eps < 1:
                raise ArithmeticError("eps should be between in (0, 1]")
            self.eps = eps
        if tol is None:
            self.tol = [1.0e-05, sqrt(np.finfo(float).eps)]
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
    @staticmethod
    def get_one_dimensional():
        return [Model.E, Model.V]

    @staticmethod
    def get_multi_dimensional():
        return [Model.EII, Model.VII, Model.EEI, Model.VEI,
                Model.EVI, Model.VVI, Model.EEE, Model.EVE,
                Model.VEE, Model.VVE, Model.EEV, Model.VEV,
                Model.EVV, Model.VVV]

    @staticmethod
    def get_less_observations():
        return [Model.EEE, Model.EEV, Model.VEV, Model.VVV]

