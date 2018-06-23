import numpy as np
import warnings

from mclust.Exceptions import ModelError
from mclust.Control import EMControl
from mclust.Models import Model, MixtureModel
from mclust.Utility import round_sig
from mclust.fortran import mclust
from mclust.MVN import MVNX
from mclust.variance import *

# TODO implement vinv and prior


class ME(MixtureModel):
    def __init__(self, control=EMControl()):
        super().__init__()
        self._set_control(control)
        self.z = None
        self.vinv = None

    def _set_control(self, control=EMControl()):
        self.iterations = np.array(control.itmax[0], order='F')
        self.err = np.array(control.tol[0], order='F')
        self.loglik = np.array(control.eps, order='F')
        self.control = control

    def fit(self, data, z, prior=None, control=None, vinv=None):
        """
        Runs ME algorithm on data starting with probabilities defined in z

        :param data: data to run ME algorithm on
        :param z: matrix containing the probability of data observation i belonging
                  to group j
        :param prior: optional prior
        :param control: MEControl object with control parameters for ME algorithm
        :param vinv: An estimate of the reciprocal hypervolume of the data region,
               when the model is to include a noise term. Set to a negative value
               or zero if a noise term is desired, but an estimate is unavailable
               — in that case function hypvol will be used to obtain the estimate.
               The default is not to assume a noise term in the model through the
               setting Vinv=None.
        :return: return code of Algorithm
        """
        if control is not None:
            self._set_control(control)
        pass

    def _check_output(self):
        if self.loglik > round_sig(np.finfo(float).max, 6) or self.loglik == float('nan'):
            warnings.warn("singular covariance")
            self.mean = self.pro = self.variance = self.z = self.loglik = float('nan')
            return -1
        if self.loglik < -round_sig(np.finfo(float).max, 6):
            if self.control.equalPro:
                warnings.warn("z column fell below threshold")
            else:
                warnings.warn("mixing proporting fell below threshold")
            self.mean = self.pro = self.variance = self.z = self.loglik = float('nan')
            return -2 if self.control.equalPro else -3
        if self.iterations >= self.control.itmax[0]:
            warnings.warn("iteration limit reached")
            self.iterations = -self.iterations
            return 1
        return 0


class MEE(ME):
    def __init__(self):
        super().__init__()
        self.model = Model.E

    def fit(self, data, z, prior=None, control=None, vinv=None):
        super().fit(data, z, prior, control, vinv)

        if data.ndim != 1:
            raise ModelError("data must be 1 dimensional")
        self.d = 1
        n = len(data)
        self.n = n
        if z.shape[0] != n:
            raise ModelError("row dimension of z should be equal length of the data")

        K = z.shape[1]
        G = K
        if vinv is not None:
            G = K - 1
            if vinv <= 0:
                vinv = hypvol(data, reciprocal=True)
        self.G = G
        self.vinv = vinv

        if np.all(z == None):
            warnings.warn("z is missing")
            self.variance = None
            self.pro = np.repeat([None], G)
            self.mean = np.repeat([None], G)
            return 9

        if np.any(z == None) or np.any(z < 0) or np.any(z > 1):
            raise ModelError("improper specification of z")

        self.mean = np.zeros(G, float, order='F')
        sigmasq = np.array(1, float, order='F')
        self.pro = np.zeros(K, float, order='F')
        self.z = z
        if prior is None:
            mclust.me1e(self.control.equalPro,
                        data,
                        G,
                        -1.0 if vinv is None else vinv,
                        self.z,
                        self.iterations,
                        self.err,
                        self.loglik,
                        self.mean,
                        sigmasq,
                        self.pro
                        )
        else:
            raise NotImplementedError("prior not yet supported")

        if np.any(sigmasq <= max(self.control.eps, 0)):
            warnings.warn("sigma-squared falls below threshold")
            self.mean = self.pro = self.variance = self.z = self.loglik = float('nan')
            return -1
        self.variance = VarianceSigmasq(self.d, self.G, sigmasq)

        return self._check_output()


class MEV(ME):
    def __init__(self):
        super().__init__()
        self.model = Model.V

    def fit(self, data, z, prior=None, control=None, vinv=None):
        super().fit(data, z, prior, control, vinv)

        # potential FIXME most of this code is the same to MEE.run
        if data.ndim != 1:
            raise ModelError("data must be 1 dimensional")
        self.d = 1
        n = len(data)
        self.n = n
        if z.shape[0] != n:
            raise ModelError("row dimension of z should be equal length of the data")

        K = z.shape[1]
        G = K
        if vinv is not None:
            G = K - 1
            if vinv <= 0:
                vinv = hypvol(data, reciprocal=True)
        self.G = G
        self.vinv = vinv

        if np.all(z == None):
            warnings.warn("z is missing")
            self.variance = None
            self.pro = np.repeat([None], G)
            self.mean = np.repeat([None], G)
            return 9

        if np.any(z == None) or np.any(z < 0) or np.any(z > 1):
            raise ModelError("improper specification of z")

        self.mean = np.zeros(G, float, order='F')
        sigmasq = np.zeros(G, float, order='F')
        self.pro = np.zeros(K, float, order='F')
        self.z = z

        if prior is None:
            mclust.me1v(self.control.equalPro,
                        data,
                        G,
                        -1.0 if vinv is None else vinv,
                        self.z,
                        self.iterations,
                        self.err,
                        self.loglik,
                        self.mean,
                        sigmasq,
                        self.pro
                        )
        else:
            raise NotImplementedError("prior not yet supported")

        if np.any(sigmasq <= max(self.control.eps, 0)):
            warnings.warn("sigma-squared falls below threshold")
            self.mean = self.pro = self.variance = self.z = self.loglik = float('nan')
            return -1
        self.variance = VarianceSigmasq(self.d, self.G, sigmasq)

        return self._check_output()


class MEX(ME):
    def __init__(self):
        super().__init__()
        self.model = Model.X

    def fit(self, data, z, prior=None, control=None, vinv=None):
        if data.ndim != 1:
            raise ModelError("data must be 1 dimensional")
        self.d = 1
        n = len(data)
        self.n = n
        if z.shape[0] != n:
            raise ModelError("row dimension of z should be equal length of the data")

        self.G = z.shape[1]
        self.z = z

        mvn = MVNX()
        self.prior = prior
        self.returnCode = mvn.fit(data, prior)
        self.loglik = mvn.loglik
        self.mean = mvn.mean
        self.variance = mvn.variance
        self.pro = mvn.pro
        return self.returnCode


class MEVVV(ME):
    def __init__(self):
        super().__init__()
        self.model = Model.VVV

    def fit(self, data, z, prior=None, control=None, vinv=None):
        super().fit(data, z, prior, control, vinv)

        if data.ndim != 2:
            raise ModelError("data must be 2 dimensional")
        self.d = data.shape[1]
        self.n = len(data)
        if z.shape[0] != self.n:
            raise ModelError("row dimension of z should be equal length of the data")

        K = z.shape[1]
        self.G = K
        self.vinv = vinv
        self.prior = prior

        if vinv is not None:
            self.G = K - 1
            if vinv <= 0:
                vinv = hypvol(data, reciprocal=True)

        if np.all(z == None):
            warnings.warn("z is missing")
            self.variance = None
            self.pro = np.repeat([None], self.G)
            self.mean = np.repeat([None], self.G * self.d).reshape(self.d, self.G)
            self.z = z
            return 9

        if np.any(z == None) or np.any(z < 0) or np.any(z > 1):
            raise ModelError("improper specification of z")

        self.mean = np.zeros(self.G * self.d, float).reshape(self.d, self.G, order='F')
        cholsigma = np.zeros(self.d * self.d * self.G, float).reshape(self.d, self.d, self.G, order='F')
        self.pro = np.zeros(K, float, order='F')
        w = np.zeros(self.d, float, order='F')
        s = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')
        self.z = z

        if prior is None:
            mclust.mevvv(self.control.equalPro,
                         data,
                         self.G,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         self.mean,
                         cholsigma,
                         self.pro,
                         w,
                         s,
                         )
        else:
            raise NotImplementedError()

        # cholsigma needs to be transposed as the order fortran returns does not correspond
        # to the order python expects
        cholsigma = cholsigma.transpose((2, 0, 1))

        self.variance = VarianceCholesky(self.d, self.G, cholsigma)

        return self._check_output()

def model_to_me(model):
    return {
        Model.E: MEE(),
        Model.V: MEV(),
        Model.X: MEX(),
        Model.VVV: MEVVV()
    }.get(model)
