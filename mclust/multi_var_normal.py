import sys
import warnings

from mclust.em import MEE, MEEII, MEEEI, MEEEE
from mclust.fortran import mclust
from mclust.models import Model, MixtureModel
from mclust.utility import round_sig
from mclust.variance import *


class MVN(MixtureModel):
    """Fit uni/multi-variate normal to data with specified prior"""
    def fit(self):
        self.z = np.full((self.n, 1), 1)

    def _check_output(self):
        if self.loglik > round_sig(sys.float_info.max, 6):
            warnings.warn("singular covariance")
            self.loglik = None
            self.return_code = -1
            return -1
        self.return_code = 0
        return self.return_code


class MVNX(MVN):
    def __init__(self, data, z=None, prior=None, **kwargs):
        super().__init__(data, z, prior)
        if self.data.ndim != 1:
            raise DimensionError("Data must be one dimensional, actual dimension {}".format(self.data.ndim))
        self.model = Model.X
        self.g = 1
        self.pro = np.array([1], float, order='F')

    def fit(self):
        super().fit()
        floatData = self.data.astype(float, order='F')

        if self.prior is None:
            self.mean, sigmasq, self.loglik = mclust.mvn1d(floatData)
        else:
            raise NotImplementedError()

        self.variance = VarianceSigmasq(self.d, self.g, np.array(sigmasq))
        self.mean = np.array([[self.mean]])

        return self._check_output()

    def component_density(self, new_data=None, logarithm=False):
        # density calculation is based on e-step, multivariate normals don't have an e-step
        # therefor a dummy MEE object (with e-step) is used to calculate component densities
        em_model = MEE(self.data, self.z)
        self.copy_onto(em_model)
        return em_model.component_density(new_data, logarithm=logarithm)


class MVNXII(MVN):
    def __init__(self, data, z=None, prior=None, **kwargs):
        super().__init__(data, z, prior)
        if self.data.ndim != 2:
            raise DimensionError("MVNXII requires two-dimensional data, actual dimension {}".format(self.data.ndim))
        self.model = Model.XII

    def fit(self):
        super().fit()
        self.g = 1
        self.pro = np.array([1], float, order='F')

        self.mean = np.zeros(self.d, float, order='F')
        sigmasq = np.array(0, float, order='F')
        self.loglik = np.array(0, float, order='F')

        if self.prior is None:
            data_copy = self.data.copy()
            mclust.mvnxii(data_copy, self.d, self.mean, sigmasq, self.loglik)
        else:
            raise NotImplementedError()

        self.variance = VarianceSigmasq(self.d, self.g, sigmasq)
        self.mean = np.array([self.mean])

        return self._check_output()

    def component_density(self, new_data=None, logarithm=False):
        # density calculation is based on e-step, multivariate normals don't have an e-step
        # therefor a dummy MEEII object (with e-step) is used to calculate component densities
        em_model = MEEII(self.data, self.z)
        self.copy_onto(em_model)
        return em_model.component_density(new_data, logarithm=logarithm)


class MVNXXI(MVN):
    def __init__(self, data, z=None, prior=None, **kwargs):
        super().__init__(data, z, prior)
        if self.data.ndim != 2:
            raise DimensionError("MVNXXI requires two-dimensional data, actual dimesnion {}".format(self.data.ndim))
        self.model = Model.XXI

    def fit(self):
        super().fit()
        self.g = 1
        self.pro = np.array([1], float, order='F')

        self.mean = np.zeros(self.d, float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.loglik = np.array(0, float, order='F')

        if self.prior is None:
            data_copy = self.data.copy()
            mclust.mvnxxi(data_copy, self.d, self.mean, scale, shape, self.loglik)
        else:
            raise NotImplementedError()

        self.variance = VarianceDecomposition(self.d, self.g, scale, shape)
        self.mean = np.array([self.mean])

        return self._check_output()

    def component_density(self, new_data=None, logarithm=False):
        # density calculation is based on e-step, multivariate normals don't have an e-step
        # therefor a dummy MEEEI object (with e-step) is used to calculate component densities
        em_model = MEEEI(self.data, self.z)
        self.copy_onto(em_model)
        return em_model.component_density(new_data, logarithm=logarithm)


class MVNXXX(MVN):
    def __init__(self, data, z=None, prior=None, **kwargs):
        super().__init__(data, z, prior)
        if self.data.ndim != 2:
            raise DimensionError("MVNXXX requires two-dimensional data, actual dimension: {}".format(self.data.ndim))
        self.model = Model.XXX

    def fit(self):
        super().fit()
        self.g = 1
        self.pro = np.array([1], float, order='F')

        self.mean = np.zeros(self.d, float, order='F')
        cholsigma = np.zeros(self.d * self.d).reshape(self.d, self.d, order='F')
        self.loglik = np.array(0, float, order='F')

        if self.prior is None:
            data_copy = self.data.copy()
            mclust.mvnxxx(data_copy, self.mean, cholsigma, self.loglik)
        else:
            raise NotImplementedError()

        self.variance = VarianceCholesky(self.d, self.g, np.array([cholsigma]))
        self.mean = np.array([self.mean])

        return self._check_output()

    def component_density(self, new_data=None, logarithm=False):
        # density calculation is based on e-step, multivariate normals don't have an e-step
        # therefor a dummy MEEEE object (with e-step) is used to calculate component densities
        em_model = MEEEE(self.data, self.z)
        self.copy_onto(em_model)
        return em_model.component_density(new_data, logarithm=logarithm)
