import copy
import warnings
from math import sqrt

import numpy as np

from mclust.control import EMControl
from mclust.exceptions import ModelError, AbstractMethodError
from mclust.fortran import mclust, mclustaddson
from mclust.models import Model, MixtureModel
from mclust.utility import round_sig, mclust_map
from mclust.variance import VarianceSigmasq, VarianceCholesky, VarianceDecomposition


# TODO implement vinv and prior

class ME(MixtureModel):
    """
    MixtureModel that utilises EM algorithm to fit model to the data.
    """
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior)
        self._set_control(control)
        self.vinv = None

    def _set_control(self, control=EMControl()):
        """
        Initialise properties set by EMControl object
        :param control: EMControl object
        """
        self.iterations = np.array(control.itmax[0], order='F')
        self.err = np.array(control.tol[0], order='F')
        self.loglik = np.array(control.eps, order='F')
        self.control = control

    def fit(self, control=None, vinv=None):
        if control is not None:
            self._set_control(control)

        self.return_code = self._check_z_matrix()
        if self.return_code != 0:
            return self.return_code

        g = self.z.shape[1]
        if vinv is not None:
            g = self.z.shape[1] - 1
            if vinv <= 0:
                vinv = hypvol(self.data, reciprocal=True)
        self.g = g
        self.vinv = vinv

        self._me_fortran(control, vinv)
        self.return_code = self._handle_output()
        return self.return_code

    def m_step(self):
        """
        Executes single maximization step of EM algorithms on this mixture model.
        :return: return code indicating success, see return_code in MixtureModel.
        """
        if self.g is None:
            self.g = self.z.shape[1]

    def e_step(self):
        """
        Executes single expectation step of EM algorithms on this mixture model.
        :return: return code indicating success, see return_code in MixtureModel.
        """
        raise AbstractMethodError()

    def _check_z_matrix(self):
        """
        Check if current z matrix is valid and does not have missing values.
        :return:
        """
        if self.z is None:
            warnings.warn("z is missing")
            return 9
        if self.z.shape[0] != self.n:
            raise ModelError("row dimension of z should be equal length of the data")

        if np.all(self.z == None):
            warnings.warn("z is missing")
            self.variance = None
            self.pro = np.repeat([None], self.g)
            self.mean = np.repeat([None], self.g)
            return 9

        if np.any(self.z == None) or np.any(self.z < 0) or np.any(self.z > 1):
            raise ModelError("improper specification of z")

        return 0

    def _me_fortran(self, control, vinv):
        """
        Method that calls Fortran code to fit model, implemented by children.
        """
        raise AbstractMethodError()

    def _m_step_fortran(self):
        """
        Method that calls Fortran code to run M-step, implemented by children.
        """
        raise AbstractMethodError()

    def _handle_output(self):
        """
        Checks properties of MixtureModel after fitting
        :return: return code, see return_code MixtureModel
        """
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
        if np.any(np.isinf(self.variance.get_covariance())):
            warnings.warn("singular covariance")
            self.variance = self.mean = self.pro = self.loglik = None
            return -1
        if self.iterations >= self.control.itmax[0]:
            warnings.warn("iteration limit reached")
            self.iterations = -self.iterations
            return 1
        return 0

    def classify(self):
        result = super().classify()
        if result is not None:
            return result
        if pow((np.sum(self.pro) - np.sum(np.mean(self.z, axis=0))), 2) > sqrt(np.finfo(float).eps):
            print("pro and z mean condition thingy holds")
            self.m_step()

        return mclust_map(self.z)

    def component_density(self, new_data=None, logarithm=False):
        model = copy.deepcopy(self)
        if new_data is not None:
            model.data = new_data
            model.n = new_data.shape[0]

        if self._check_parameters() != 0:
            return None

        model.pro = np.array([-1], float, order='F')
        model.e_step()

        if model.return_code or model.return_code == -9:
            return None
        elif model.return_code == 0:
            if not logarithm:
                model.z = np.exp(model.z)
        self.return_code = model.return_code
        return model.z

    def _check_parameters(self):
        """
        Checks if pro, mean and variance are missing or contain missing values.
        :return: return code, see return_code MixtureModel.
        """
        if self.pro is None or np.any(np.isnan(self.pro)) or \
                self.mean is None or np.any(np.isnan(self.mean)) or \
                self.variance is None or np.any(np.isnan(self.variance.get_covariance())):
            warnings.warn("parameters are missing")
            self.z = np.full((self.n, self.d), float('nan'))
            self.loglik = float('nan')
            self.return_code = 9
            return 9
        return 0


class ME1Dimensional(ME):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        if self.data.ndim != 1:
            raise ModelError("data must be 1 dimensional")
        self.sigmasq = None

    def _handle_output(self):
        if np.any(self.sigmasq <= max(self.control.eps, 0)):
            warnings.warn("sigma-squared falls below threshold")
            self.mean = self.pro = self.variance = self.z = self.loglik = float('nan')
            return -1
        return super()._handle_output()

    def m_step(self):
        super().m_step()
        ret = self._check_z_matrix()
        if ret != 0:
            return ret

        self._m_step_fortran()

        if np.any(self.sigmasq > round_sig(np.finfo(float).max, 6)):
            warnings.warn("cannot compute M-step")
            self.pro = self.mean = self.sigmasq = float("nan")
            self.return_code = -1
        self.variance = VarianceSigmasq(self.d, self.g, self.sigmasq)
        self.mean = np.array([self.mean]).transpose()
        self.return_code = 0
        return 0


class MEE(ME1Dimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.E

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.g, float, order='F')
        self.sigmasq = np.array(1, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.me1e(self.control.equalPro,
                        self.data,
                        self.g,
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

        self.variance = VarianceSigmasq(self.d, self.g, self.sigmasq)
        self.mean = np.array([self.mean]).transpose()

    def _m_step_fortran(self):
        self.mean = np.zeros(self.g, float, order='F')
        self.sigmasq = np.array(1, float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.ms1e(self.data,
                        self.z,
                        self.g,
                        self.mean,
                        self.sigmasq,
                        self.pro)
        else:
            raise NotImplementedError("prior not yet supported")

    def e_step(self):
        if self._check_parameters() != 0:
            return self.return_code

        if not isinstance(self.variance, VarianceSigmasq):
            warnings.warn("incorrect variance parameter")
            self.return_code = -1
            return self.return_code

        if self.pro[0] != -1:
            self.pro = self.pro / np.sum(self.pro)

            # TODO implement vinv/noise
            noise = len(self.pro) == self.g + 1
            if not noise:
                if len(self.pro) != self.g:
                    raise ModelError("pro impoperly specified")
                k = self.g
            else:
                k = self.g + 1
        else:
            k = self.g

        self.z = np.zeros((self.n, k), float, order='F')
        mclust.es1e(self.data,
                    self.mean.transpose().flatten(),
                    self.variance.sigmasq[0],
                    self.pro,
                    self.g,
                    -1.0 if self.vinv is None else self.vinv,
                    self.loglik,
                    self.z)

        if self.loglik > round_sig(np.finfo(float).max, 6):
            warnings.warn("cannot compute E-step")
            self.loglik = None
            self.return_code = -1
        else:
            self.return_code = 0
        return self.return_code


class MEV(ME1Dimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.V

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.g, float, order='F')
        self.sigmasq = np.zeros(self.g, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')

        if self.prior is None:
            mclust.me1v(self.control.equalPro,
                        self.data,
                        self.g,
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

        self.variance = VarianceSigmasq(self.d, self.g, self.sigmasq)
        self.mean = np.array([self.mean]).transpose()

    def _m_step_fortran(self):
        self.mean = np.zeros(self.g, float, order='F')
        self.sigmasq = np.zeros(self.g, float, order='F')
        self.pro = np.zeros(self.g, float, order='F')

        if self.prior is None:
            mclust.ms1v(self.data,
                        self.z,
                        self.g,
                        self.mean,
                        self.sigmasq,
                        self.pro)
        else:
            raise NotImplementedError("prior not yet supported")

    def e_step(self):
        if self._check_parameters() != 0:
            return self.return_code

        if not isinstance(self.variance, VarianceSigmasq):
            warnings.warn("incorrect variance parameter")
            self.return_code = -1
            return self.return_code

        if self.pro[0] != -1:
            self.pro = self.pro / np.sum(self.pro)

            # TODO implement vinv/noise
            noise = len(self.pro) == self.g + 1
            if not noise:
                if len(self.pro) != self.g:
                    raise ModelError("pro impoperly specified")
                k = self.g
            else:
                k = self.g + 1
        else:
            k = self.g

        self.z = np.zeros((self.n, k), float, order='F')
        mclust.es1v(self.data,
                    self.mean.transpose().flatten(),
                    self.variance.sigmasq,
                    self.pro,
                    self.g,
                    -1.0 if self.vinv is None else self.vinv,
                    self.loglik,
                    self.z)

        if self.loglik > round_sig(np.finfo(float).max, 6):
            warnings.warn("cannot compute E-step")
            self.loglik = None
            self.return_code = -1
        else:
            self.return_code = 0
        return self.return_code


class MEMultiDimensional(ME):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        if self.data.ndim != 2:
            raise ModelError("data must be 2 dimensional")

    def m_step(self):
        super().m_step()
        ret = self._check_z_matrix()
        if ret != 0:
            return ret

        self._m_step_fortran()

        if np.any(self.mean > round_sig(np.finfo(float).max, 6)):
            warnings.warn("cannot compute M-step")
            self.mean = self.variance = self.pro = float('nan')
            self.return_code = -1
            return -1

        self.return_code = 0

    def e_step(self):
        if self._check_parameters() != 0:
            return

        # For density estimation mixing proportions are not used
        if self.pro[0] != -1:
            self.pro = self.pro / np.sum(self.pro)

        # TODO implemenmt vinv
        k = self.g

        # not all models have cholsigma, so this needs to be calculated
        variance = self.variance.get_covariance().transpose(2, 1, 0)

        w = np.zeros(self.d, float, order='F')
        self.z = np.zeros((self.n, k), float, order='F')

        mclust.esvvv(False,
                     self.data,
                     self.mean.transpose(),
                     variance,
                     self.pro,
                     self.g,
                     -1 if self.vinv is None else self.vinv,
                     w,
                     self.loglik,
                     self.z
                     )

        info = w[0]
        if info:
            if info > 0:
                warnings.warn("sigma is not positive definite")
            else:
                warnings.warn("input error for LAPACK DPOTRF")
            self.loglik = None
            self.return_code = -9
        elif self.loglik > round_sig(np.finfo(float).max, 6):
            warnings.warn("cannot compute E-step")
            self.loglik = None
            self.return_code = -1
        else:
            self.return_code = 0

        return self.return_code


class MEEII(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EII

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.d * self.g, float).reshape(self.d, self.g, order='F')
        sigmasq = np.array(0, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.meeii(self.control.equalPro,
                         self.data,
                         self.g,
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
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceSigmasq(self.d, self.g, sigmasq)

    def _m_step_fortran(self):
        self.mean = np.zeros(self.d * self.g, float).reshape(self.d, self.g, order='F')
        sigmasq = np.array(0, float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.mseii(self.data,
                         self.z,
                         self.g,
                         self.mean,
                         sigmasq,
                         self.pro)
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceSigmasq(self.d, self.g, sigmasq)


class MEVII(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VII

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.d * self.g, float).reshape(self.d, self.g, order='F')
        sigmasq = np.zeros(self.g, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.mevii(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         self.mean,
                         sigmasq,
                         self.pro)
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceSigmasq(self.d, self.g, sigmasq)

    def _m_step_fortran(self):
        self.mean = np.zeros(self.d * self.g, float).reshape(self.d, self.g, order='F')
        sigmasq = np.zeros(self.g, float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.msvii(self.data,
                         self.z,
                         self.g,
                         self.mean,
                         sigmasq,
                         self.pro)
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceSigmasq(self.d, self.g, sigmasq)


class MEEEI(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EEI

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.d * self.g, float).reshape(self.d, self.g, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.meeei(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         self.mean,
                         scale,
                         shape,
                         self.pro
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape)

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.mseei(self.data,
                         self.z,
                         self.g,
                         self.mean,
                         scale,
                         shape,
                         self.pro
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape)


class MEVEI(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VEI
        self.iters = np.array(control.itmax, np.int32, order='F')
        self.errs = np.array(control.tol, float, order='F')

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        scl = np.zeros(self.g, float, order='F')
        shp = np.zeros(self.d, float, order='F')
        w = np.zeros((self.d, self.g), float, order='F')
        if self.prior is None:
            mclust.mevei(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iters,
                         self.errs,
                         self.loglik,
                         self.mean,
                         scale,
                         shape,
                         self.pro,
                         scl,
                         shp,
                         w
                         )
        else:
            raise NotImplementedError()

        self.iterations = self.iters[0]
        self.err = self.errs[0]
        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape)

    def _m_step_fortran(self):
        self.iterations = np.array(self.control.itmax[1], np.int32, order='F')
        self.err = np.array(self.control.tol[1], float, order='F')
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        scl = np.zeros(self.g, float, order='F')
        shp = np.zeros(self.d, float, order='F')
        w = np.zeros((self.d, self.g), float, order='F')
        if self.prior is None:
            mclust.msvei(self.data,
                         self.z,
                         self.g,
                         self.iterations,
                         self.err,
                         self.mean,
                         scale,
                         shape,
                         self.pro,
                         scl,
                         shp,
                         w
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape)

        if self.iterations >= self.control.itmax[1]:
            warnings.warn("inner iteration limit reached")

    def _handle_output(self):
        ret = super()._handle_output()
        if ret == 0 or ret == 1:
            if self.iters[1] >= self.control.itmax[0]:
                warnings.warn("inner iteration limit reached")
                self.iters[1] = -self.iters[1]
        return ret


class MEEVI(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EVI

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.meevi(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         self.mean,
                         scale,
                         shape,
                         self.pro
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose())

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.msevi(self.data,
                         self.z,
                         self.g,
                         self.mean,
                         scale,
                         shape,
                         self.pro
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose())


class MEVVI(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VVI

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        if self.prior is None:
            mclust.mevvi(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         self.mean,
                         scale,
                         shape,
                         self.pro
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose())

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.msvvi(self.data,
                         self.z,
                         self.g,
                         self.mean,
                         scale,
                         shape,
                         self.pro
                         )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose())


class MEEEE(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EEE

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.g * self.d, float).reshape(self.d, self.g, order='F')
        cholsigma = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        w = np.zeros(self.d, float, order='F')

        if self.prior is None:
            mclust.meeee(self.control.equalPro,
                         self.data,
                         self.g,
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
        self.variance = VarianceCholesky(self.d, self.g, cholsigma)

    def _m_step_fortran(self):
        w = np.zeros(self.d, float, order='F')
        self.mean = np.zeros(self.g * self.d, float).reshape(self.d, self.g, order='F')
        cholsigma = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.mseee(self.data,
                         self.z,
                         self.g,
                         w,
                         self.mean,
                         cholsigma,
                         self.pro
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        self.variance = VarianceCholesky(self.d, self.g, cholsigma)


class MEEVE(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EVE

        self.iterin = np.array(0, np.int32, order='F')
        self.iterout = np.array(0, np.int32, order='F')
        self.errin = np.array(0, float, order='F')
        self.errout = np.array(0, float, order='F')
        self.info = np.array(0, np.int32, order='F')

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        orientation = np.asfortranarray(np.identity(self.d, float))
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.ones((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')

        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')

        if self.prior is None:
            mclustaddson.meeve(self.data,
                               self.z,
                               self.mean,
                               orientation,
                               u,
                               scale,
                               shape,
                               self.pro,
                               -1.0 if vinv is None else vinv,
                               self.loglik,
                               self.control.equalPro,
                               self.control.itmax[1],
                               self.control.tol[1],
                               self.control.itmax[0],
                               self.control.tol[0],
                               self.control.eps,
                               self.iterin,
                               self.errin,
                               self.iterout,
                               self.errout,
                               lwork,
                               self.info
                               )
        else:
            raise NotImplementedError("priors are not yet supported")

        self.iterations = self.iterout
        self.err = self.errout
        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose(), orientation.transpose())

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        orientation = np.asfortranarray(np.identity(self.d, float))
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.ones((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        self.iterin = np.array(self.control.itmax[1], np.int32, order='F')
        self.errin = np.array(self.control.tol[1], np.int32, order='F')

        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        if self.prior is None:
            mclustaddson.mseve(self.data,
                               self.z,
                               self.mean,
                               u,
                               orientation,
                               scale,
                               shape,
                               self.pro,
                               lwork,
                               self.info,
                               self.control.itmax[1],
                               self.control.tol[1],
                               self.iterin,
                               self.errin,
                               self.control.eps
                               )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose(), orientation.transpose())

    def _handle_output(self):
        if self.info:
            if self.info > 0:
                warnings.warn("LAPACK DSYEV or DGESVD fails to converge")
            elif self.info < 0:
                warnings.warn("input error for LAPACK DSYEV or DGESVD")
            self.variance = self.mean = self.pro = self.loglik = None
            return -9

        ret = super()._handle_output()
        if ret == 0 or ret == 1:
            if self.iterin >= self.control.itmax[0]:
                warnings.warn("inner iteration limit reached")
                self.iterin = -self.iterin
        return ret


class MEVEE(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VEE

        self.iterin = np.array(0, np.int32, order='F')
        self.iterout = np.array(0, np.int32, order='F')
        self.errin = np.array(0, float, order='F')
        self.errout = np.array(0, float, order='F')
        self.info = np.array(0, np.int32, order='F')

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        c = np.zeros((self.d, self.d), float, order='F')
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        if self.prior is None:
            mclustaddson.mevee(self.data,
                               self.z,
                               self.mean,
                               c,
                               u,
                               scale,
                               shape,
                               self.pro,
                               -1.0 if vinv is None else vinv,
                               self.loglik,
                               self.control.equalPro,
                               self.control.itmax[1],
                               self.control.tol[1],
                               self.control.itmax[0],
                               self.control.tol[0],
                               self.control.eps,
                               self.iterin,
                               self.errin,
                               self.iterout,
                               self.errout,
                               lwork,
                               self.info
                               )
        else:
            raise NotImplementedError("priors are not yet supported")

        # TODO clean up mess with inner and outer iterations
        self.iterations = self.iterout
        self.mean = self.mean.transpose()

        if np.any(np.isnan(c)):
            orientation = c
        else:
            _, _, orientation = np.linalg.svd(c)

        self.variance = VarianceDecomposition(self.d, self.g, scale, shape, orientation.transpose())

    def _handle_output(self):
        if self.info:
            if self.info > 0:
                warnings.warn("LAPACK DSYEV or DGESVD fails to converge")
            elif self.info < 0:
                warnings.warn("input error for LAPACK DSYEV or DGESVD")
            self.variance = self.mean = self.pro = self.loglik = None
            return -9

        ret = super()._handle_output()
        if ret == 0 or ret == 1:
            if self.iterin >= self.control.itmax[0]:
                warnings.warn("inner iteration limit reached")
                self.iterin = -self.iterin
        return ret

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        c = np.zeros((self.d, self.d), float, order='F')
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.ones(self.g, float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        self.iterin = np.array(self.control.itmax[1], np.int32, order='F')
        self.errin = np.array(self.control.tol[1], np.int32, order='F')

        if self.prior is None:
            mclustaddson.msvee(self.data,
                               self.z,
                               self.mean,
                               u,
                               c,
                               scale,
                               self.pro,
                               lwork,
                               self.info,
                               self.control.itmax[1],
                               self.control.tol[1],
                               self.iterin,
                               self.errin,
                               self.control.eps
                               )
        else:
            raise NotImplementedError("prior not yet supported")

        _, shape, orientation = np.linalg.svd(c)

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape, orientation.transpose())


class MEVVE(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VVE

        self.iterin = np.array(0, np.int32, order='F')
        self.iterout = np.array(0, np.int32, order='F')
        self.errin = np.array(0, float, order='F')
        self.errout = np.array(0, float, order='F')
        self.info = np.array(0, np.int32, order='F')

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        orientation = np.asfortranarray(np.identity(self.d, float))
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.ones(self.g, float, order='F')
        shape = np.ones((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        if self.prior is None:
            mclustaddson.mevve(self.data,
                               self.z,
                               self.mean,
                               orientation,
                               u,
                               scale,
                               shape,
                               self.pro,
                               -1.0 if vinv is None else vinv,
                               self.loglik,
                               self.control.equalPro,
                               self.control.itmax[1],
                               self.control.tol[1],
                               self.control.itmax[0],
                               self.control.tol[0],
                               self.control.eps,
                               self.iterin,
                               self.errin,
                               self.iterout,
                               self.errout,
                               lwork,
                               self.info,
                               )
        else:
            raise NotImplementedError("prior not yet supported")

        self.iterations = self.iterout
        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose(), orientation.transpose())

    def _handle_output(self):
        if self.info:
            if self.info > 0:
                warnings.warn("LAPACK DSYEV or DGESVD fails to converge")
            elif self.info < 0:
                warnings.warn("input error for LAPACK DSYEV or DGESVD")
            self.variance = self.mean = self.pro = self.loglik = None
            return -9

        ret = super()._handle_output()
        if ret == 0 or ret == 1:
            if self.iterin >= self.control.itmax[0]:
                warnings.warn("inner iteration limit reached")
                self.iterin = -self.iterin
        return ret

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        orientation = np.asfortranarray(np.identity(self.d, float))
        scale = np.ones(self.g, float, order='F')
        shape = np.ones((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        self.iterin = np.array(self.control.itmax[1], np.int32, order='F')
        self.errin = np.array(self.control.tol[1], np.int32, order='F')

        if self.prior is None:
            mclustaddson.msvve(self.data,
                               self.z,
                               self.mean,
                               u,
                               orientation,
                               scale,
                               shape,
                               self.pro,
                               lwork,
                               self.info,
                               self.control.itmax[1],
                               self.control.tol[1],
                               self.iterin,
                               self.errin,
                               self.control.eps
                               )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose(), orientation.transpose())


class MEEEV(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EEV

    def _me_fortran(self, control, vinv):
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        orientation = np.zeros((self.d, self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        w = np.zeros(lwork, float, order='F')
        s = np.zeros(self.d, float, order='F')
        if self.prior is None:
            mclust.meeev(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iterations,
                         self.err,
                         self.loglik,
                         lwork,
                         self.mean,
                         scale,
                         shape,
                         orientation,
                         self.pro,
                         w,
                         s
                         )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape, orientation.transpose((2, 1, 0)))

    def _m_step_fortran(self):
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.array(0, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        orientation = np.zeros((self.d, self.d, self.g), float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        w = np.zeros(lwork, float, order='F')
        if self.prior is None:
            mclust.mseev(self.data,
                         self.z,
                         self.g,
                         w,
                         lwork,
                         self.mean,
                         scale,
                         shape,
                         orientation,
                         self.pro
                         )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape, orientation.transpose((2, 1, 0)))


class MEVEV(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VEV

        self.iters = np.array(control.itmax, np.int32, order='F')
        self.errs = np.array(control.tol, float, order='F')
        self.info = None

    def _me_fortran(self, control, vinv):
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        orientation = np.zeros((self.d, self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        w = np.zeros(lwork, float, order='F')
        s = np.zeros(self.d, float, order='F')
        if self.prior is None:
            mclust.mevev(self.control.equalPro,
                         self.data,
                         self.g,
                         -1.0 if vinv is None else vinv,
                         self.z,
                         self.iters,
                         self.errs,
                         self.loglik,
                         lwork,
                         self.mean,
                         scale,
                         shape,
                         orientation,
                         self.pro,
                         w,
                         s
                         )
        else:
            raise NotImplementedError("prior not yet supported")

        self.iterations = self.iters[0]
        self.info = lwork
        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape, orientation.transpose((2, 1, 0)))

    def _handle_output(self):
        if self.info:
            if self.info > 0:
                warnings.warn("LAPACK DSYEV or DGESVD fails to converge")
            elif self.info < 0:
                warnings.warn("input error for LAPACK DSYEV or DGESVD")
            self.variance = self.mean = self.pro = self.loglik = None
            return -9

        ret = super()._handle_output()
        if ret == 0 or ret == 1:
            if self.iters[1] >= self.control.itmax[0]:
                warnings.warn("inner iteration limit reached")
                self.iters[1] = -self.iters[1]
        return ret

    def _m_step_fortran(self):
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        w = np.zeros(lwork, float, order='F')
        self.iterations = self.control.itmax[1]
        self.err = self.control.tol[1]
        self.mean = np.zeros((self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros(self.d, float, order='F')
        orientation = np.zeros((self.d, self.d, self.g), float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        if self.prior is None:
            mclust.msvev(self.data,
                         self.z,
                         self.g,
                         w,
                         lwork,
                         self.iterations,
                         self.err,
                         self.mean,
                         scale,
                         shape,
                         orientation,
                         self.pro)
        else:
            raise NotImplementedError("prior not yet supported")

        self.info = lwork
        self.mean = self.mean.transpose()
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape, orientation.transpose((2, 1, 0)))


class MEEVV(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.EVV

        self.info = np.array(0, np.int32, order='F')

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        orientation = np.zeros((self.d, self.d, self.g), float, order='F')
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        self.loglik = np.array(0, float, order='F')
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')

        if self.prior is None:
            mclustaddson.meevv(self.data,
                               self.z,
                               self.mean,
                               orientation,
                               u,
                               scale,
                               shape,
                               self.pro,
                               -1.0 if vinv is None else vinv,
                               self.loglik,
                               self.control.equalPro,
                               self.control.itmax[0],
                               self.control.tol[0],
                               self.control.eps,
                               self.iterations,
                               self.err,
                               lwork,
                               self.info
                               )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        scale = np.array(scale[0], float, order='F')
        self.variance = VarianceDecomposition(self.d, self.g, scale,
                                              shape.transpose(),
                                              orientation.transpose((2, 1, 0)))

    def _handle_output(self):
        if self.info:
            if self.info > 0:
                warnings.warn("LAPACK DSYEV or DGESVD fails to converge")
            elif self.info < 0:
                warnings.warn("input error for LAPACK DSYEV or DGESVD")
            self.variance = self.mean = self.pro = self.loglik = None
            return -9
        return super()._handle_output()

    def _m_step_fortran(self):
        self.mean = np.zeros((self.d, self.g), float, order='F')
        orientation = np.zeros((self.d, self.d, self.g), float, order='F')
        u = np.zeros((self.d, self.d, self.g), float, order='F')
        scale = np.zeros(self.g, float, order='F')
        shape = np.zeros((self.d, self.g), float, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        lwork = np.array(max(3 * min(self.n, self.d) + max(self.n, self.d),
                             5 * min(self.n, self.d),
                             self.d + self.g), np.int32, order='F')
        self.info = np.array(0, np.int32, order='F')
        if self.prior is None:
            mclustaddson.msevv(self.data,
                               self.z,
                               self.mean,
                               orientation,
                               u,
                               scale,
                               shape,
                               self.pro,
                               lwork,
                               self.info,
                               self.control.eps
                               )
        else:
            raise NotImplementedError("prior not yet supported")

        self.mean = self.mean.transpose()
        scale = np.array(scale[0], float, order='F')
        self.variance = VarianceDecomposition(self.d, self.g, scale, shape.transpose(),
                                              orientation.transpose((2, 1, 0)))


class MEVVV(MEMultiDimensional):
    def __init__(self, data, z, prior=None, control=EMControl()):
        super().__init__(data, z, prior, control)
        self.model = Model.VVV

    def _me_fortran(self, control, vinv):
        self.mean = np.zeros(self.g * self.d, float).reshape(self.d, self.g, order='F')
        cholsigma = np.zeros(self.d * self.d * self.g, float).reshape(self.d, self.d, self.g, order='F')
        self.pro = np.zeros(self.z.shape[1], float, order='F')
        w = np.zeros(self.d, float, order='F')
        s = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')

        if self.prior is None:
            mclust.mevvv(self.control.equalPro,
                         self.data,
                         self.g,
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
        self.variance = VarianceCholesky(self.d, self.g, cholsigma)

    def _m_step_fortran(self):
        w = np.zeros(self.d, float, order='F')
        self.mean = np.zeros(self.g * self.d, float).reshape(self.d, self.g, order='F')
        cholsigma = np.zeros(self.d * self.d * self.g, float).reshape(self.d, self.d, self.g, order='F')
        self.pro = np.zeros(self.g, float, order='F')
        s = np.zeros(self.d * self.d, float).reshape(self.d, self.d, order='F')

        if self.prior is None:
            mclust.msvvv(self.data,
                         self.z,
                         self.g,
                         w,
                         self.mean,
                         cholsigma,
                         self.pro,
                         s
                         )
        else:
            raise NotImplementedError()

        self.mean = self.mean.transpose()
        cholsigma = cholsigma.transpose((2, 0, 1))
        self.variance = VarianceCholesky(self.d, self.g, cholsigma)
