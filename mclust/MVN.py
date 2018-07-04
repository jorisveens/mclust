from mclust.fortran import mclust

from mclust.Exceptions import *
from mclust.Models import Model, MixtureModel
from mclust.Utility import round_sig
from mclust.variance import *

import numpy as np
import sys
import warnings


class MVN(MixtureModel):
    """Fit uni/multi-variate normal to data with specified prior"""
    def fit(self):
        pass

    def _check_output(self):
        if self.loglik > round_sig(sys.float_info.max, 6):
            warnings.warn("singular covariance")
            self.loglik = None
            self.returnCode = -1
            return -1
        self.returnCode = 0
        return 0


class MVNX(MVN):
    def __init__(self, data, prior=None):
        super().__init__(data, prior)
        if self.data.ndim != 1:
            raise DimensionError(f"Data must be one dimensional, actual dimension {self.data.ndim}")
        self.model = Model.X
        self.G = 1
        self.pro = 1

    def fit(self):
        floatData = self.data.astype(float, order='F')

        if self.prior is None:
            self.mean, sigmasq, self.loglik = mclust.mvn1d(floatData)
        else:
            raise NotImplementedError()

        self.variance = VarianceSigmasq(self.d, self.G, sigmasq)
        self.mean = np.array(self.mean).transpose()

        return self._check_output()


class MVNXII(MVN):
    def __init__(self, data, prior=None):
        super().__init__(data, prior)
        if self.data.ndim != 2:
            raise DimensionError(f"MVNXII requires two-dimensional data, actual dimension {self.data.ndim}")
        self.model = Model.XII

    def fit(self):
        self.G = 1
        self.pro = 1

        self.mean = np.zeros(self.d, float, order='F')
        sigmasq = np.array(0, float, order='F')
        self.loglik = np.array(0, float, order='F')

        if self.prior is None:
            data_copy = self.data.copy()
            mclust.mvnxii(data_copy, self.d, self.mean, sigmasq, self.loglik)
        else:
            raise NotImplementedError()

        self.variance = VarianceSigmasq(self.d, self.G, sigmasq)
        self.mean = np.array([self.mean]).transpose()

        return self._check_output()


class MVNXXI(MVN):
    def __init__(self, data, prior=None):
        super().__init__(data, prior)
        if self.data.ndim != 2:
            raise DimensionError(f"MVNXXI requires two-dimensional data, actual dimesnion {self.data.ndim}")
        self.model = Model.XXI

    def fit(self):
        self.G = 1
        self.pro = 1

        self.mean = np.zeros(self.d, float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.loglik = np.array(0, float, order='F')

        if self.prior is None:
            data_copy = self.data.copy()
            mclust.mvnxxi(data_copy, self.d, self.mean, scale, shape, self.loglik)
        else:
            raise NotImplementedError()

        shape = np.identity(self.d, float) * shape
        self.variance = VarianceDecomposition(self.d, self.G, scale, shape)
        self.mean = np.array([self.mean]).transpose()

        return self._check_output()


class MVNXXX(MVN):
    def __init__(self, data, prior=None):
        super().__init__(data, prior)
        if self.data.ndim != 2:
            raise DimensionError(f"MVNXXX requires two-dimensional data, actual dimension: {self.data.ndim}")
        self.model = Model.XXX

    def fit(self):
        self.G = 1
        self.pro = 1

        self.mean = np.zeros(self.d, float, order='F')
        cholsigma = np.zeros(self.d * self.d).reshape(self.d, self.d, order='F')
        self.loglik = np.array(0, float, order='F')

        if self.prior is None:
            data_copy = self.data.copy()
            mclust.mvnxxx(data_copy, self.mean, cholsigma, self.loglik)
        else:
            raise NotImplementedError()

        self.variance = VarianceCholesky(self.d, self.G, np.array([cholsigma]))
        self.mean = np.array([self.mean]).transpose()

        return self._check_output()


def model_to_mvn(model, data, prior=None):
    print(f"model: {model}, dim: {data.ndim}")
    mod = {
        Model.X: MVNX,
        Model.E: MVNX,
        Model.V: MVNX,
        Model.EII: MVNXII,
        Model.VVV: MVNXXX,
        Model.EEE: MVNXXX
    }.get(model)
    return mod(data, prior)

