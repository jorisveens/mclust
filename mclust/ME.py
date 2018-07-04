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
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior)
        self._set_control(control)
        self.vinv = None

    def _set_control(self, control=EMControl()):
        self.iterations = np.array(control.itmax[0], order='F')
        self.err = np.array(control.tol[0], order='F')
        self.loglik = np.array(control.eps, order='F')
        self.control = control

    def fit(self, z, control=None, vinv=None):
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
        ret = self._handle_input(z, control, vinv)
        if ret != 0:
            return ret
        self.me_fortran(control, vinv)
        self.returnCode = self._handle_output()
        return self.returnCode

    def _handle_input(self, z, control, vinv):
        if control is not None:
            self._set_control(control)

        if z.shape[0] != self.n:
            raise ModelError("row dimension of z should be equal length of the data")

        G = z.shape[1]
        if vinv is not None:
            G = z.shape[1] - 1
            if vinv <= 0:
                vinv = hypvol(self.data, reciprocal=True)
        self.z = z
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

        return 0

    def me_fortran(self, control, vinv):
        pass

    def _handle_output(self):
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


class ME1Dimensional(ME):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        if self.data.ndim != 1:
            raise ModelError("data must be 1 dimensional")
        self.sigmasq = None

    def _handle_output(self):
        if np.any(self.sigmasq <= max(self.control.eps, 0)):
            warnings.warn("sigma-squared falls below threshold")
            self.mean = self.pro = self.variance = self.z = self.loglik = float('nan')
            return -1
        return super()._handle_output()


class MEE(ME1Dimensional):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        self.model = Model.E

    def me_fortran(self, control, vinv):
        self.mean = np.zeros(self.G, float, order='F')
        self.sigmasq = np.array(1, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.me1e(self.control.equalPro,
                        self.data,
                        self.G,
                        -1.0 if vinv is None else vinv,
                        self.z,
                        self.iterations,
                        self.err,
                        self.loglik,
                        self.mean,
                        self.sigmasq,
                        self.pro
                        )
        else:
            raise NotImplementedError("prior not yet supported")

        self.variance = VarianceSigmasq(self.d, self.G, self.sigmasq)
        self.mean = np.array([self.mean]).transpose()


class MEV(ME1Dimensional):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        self.model = Model.V

    def me_fortran(self, control, vinv):
        self.mean = np.zeros(self.G, float, order='F')
        self.sigmasq = np.zeros(self.G, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')

        if self.prior is None:
            mclust.me1v(self.control.equalPro,
                        self.data,
                        self.G,
                        -1.0 if vinv is None else vinv,
                        self.z,
                        self.iterations,
                        self.err,
                        self.loglik,
                        self.mean,
                        self.sigmasq,
                        self.pro
                        )
        else:
            raise NotImplementedError("prior not yet supported")

        self.variance = VarianceSigmasq(self.d, self.G, self.sigmasq)
        self.mean = np.array([self.mean]).transpose()


class MEX(ME):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        if self.data.ndim != 1:
            raise ModelError("data must be 1 dimensional")
        self.model = Model.X

    def fit(self, z, control=None, vinv=None):
        if z.shape[0] != self.n:
            raise ModelError("row dimension of z should be equal length of the data")

        self.G = z.shape[1]

        mvn = MVNX(self.data, self.prior)
        self.prior = self.prior
        self.returnCode = mvn.fit()
        self.loglik = mvn.loglik
        self.mean = mvn.mean
        self.variance = mvn.variance
        self.pro = mvn.pro
        return self.returnCode


class MEMultiDimensional(ME):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        if self.data.ndim != 2:
            raise ModelError("data must be 2 dimensional")


class MEEEE(MEMultiDimensional):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        self.model = Model.EEE

    def me_fortran(self,  control, vinv):
        self.mean = np.zeros(self.G * self.d, float).reshape(self.d, self.G, order='F')
        cholsigma = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        w = np.zeros(self.d, float, order='F')

        if self.prior is None:
            mclust.meeee(self.control.equalPro,
                         self.data,
                         self.G,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         self.mean,
                         cholsigma,
                         self.pro,
                         w
                         )
        else:
            return NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceCholesky(self.d, self.G, np.array([cholsigma]))


class MEVVV(MEMultiDimensional):
    def __init__(self, data, prior=None, control=EMControl()):
        super().__init__(data, prior, control)
        self.model = Model.VVV

    def me_fortran(self, control, vinv):
        self.mean = np.zeros(self.G * self.d, float).reshape(self.d, self.G, order='F')
        cholsigma = np.zeros(self.d * self.d * self.G, float).reshape(self.d, self.d, self.G, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        w = np.zeros(self.d, float, order='F')
        s = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')

        if self.prior is None:
            mclust.mevvv(self.control.equalPro,
                         self.data,
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
        self.mean = self.mean.transpose()
        cholsigma = cholsigma.transpose((2, 0, 1))

        self.variance = VarianceCholesky(self.d, self.G, cholsigma)


def model_to_me(model, data, prior=None, control=EMControl()):
    mod = {
        Model.E: MEE,
        Model.V: MEV,
        Model.X: MEX,
        Model.VVV: MEVVV,
        model.EEE: MEEEE
    }.get(model)
    return mod(data, prior, control)

