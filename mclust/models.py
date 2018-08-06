import copy
import numpy as np
from enum import Enum
from math import log

from mclust.exceptions import ModelError, AbstractMethodError


class Model(Enum):
    """
    Enumeration of all mclust model variants.


    Model variants stem from the eigen-decomposition of the multivariate normal covariance parameter used in Gaussian
    finite mixture modeling. Resulting in a decomposition into scale (determines volume of cluster), shape (determines
    scatter of points in each dimension
    along the eigenvectors of the covariance), and orientation (eigenvectors of covariance).

    Between different cluster groups the scale, shape and orientation can vary (V), be equal (E) or be Identity (I). Some
    combinations are redundant leading to 2 one dimensional models, and 14 multivariate models, listed in this
    enumeration. 4 more models are defined, X, XII, XXI, and XXX, which are used for models with only 1 cluster
    component.
    """
    E = 'E'
    V = 'V'
    EII = 'EII'
    VII = 'VII'
    EEI = 'EEI'
    VEI = 'VEI'
    EVI = 'EVI'
    VVI = 'VVI'
    EEE = 'EEE'
    EVE = 'EVE'
    VEE = 'VEE'
    VVE = 'VVE'
    EEV = 'EEV'
    VEV = 'VEV'
    EVV = 'EVV'
    VVV = 'VVV'

    # Mainly used for 1 group models (multivariate normal)
    X = 'X'
    XII = 'XII'
    XXI = 'XXI'
    XXX = 'XXX'

    def n_var_params(self, d, g):
        """
        Gives the number of variance parameters for parameterizations of the model.

        :param d: number of dimensions in the data
        :param g: number of cluster component
        :return: The number of variance parameters in the corresponding Gaussian mixture model.
        """
        return {
            Model.E: 1,
            Model.X: 1,
            Model.V: g,
            Model.EII: 1,
            Model.XII: 1,
            Model.VII: g,
            Model.EEI: d,
            Model.XXI: d,
            Model.VEI: g + (d - 1),
            Model.EVI: 1 + g * (d - 1),
            Model.VVI: g * d,
            Model.EEE: d * (d + 1) / 2,
            Model.XXX: d * (d + 1) / 2,
            Model.EVE: 1 + g * (d - 1) + d * (d - 1) / 2,
            Model.VEE: g + (d - 1) + d * (d - 1) / 2,
            Model.VVE: g + g * (d - 1) + d * (d - 1) / 2,
            Model.EEV: 1 + (d - 1) + g * d * (d - 1) / 2,
            Model.VEV: g + (d - 1) + g * d * (d - 1) / 2,
            Model.EVV: 1 - g + g * d * (d + 1) / 2,
            Model.VVV: g * d * (d + 1) / 2,
        }.get(self)

    def n_mclust_params(self, d, g, noise=False, equalpro=False):
        """
        Gives the number of estimated parameters for parameterizations of the model.

        :param d: number of dimensions in the data
        :param g: number of cluster components
        :param noise: A logical variable indicating whether or not the model includes an optional Poisson noise
        component.
        :param equalpro: A logical variable indicating whether or not the components in the model are assumed to be
        present in equal proportion.
        :return: The number of variance parameters in the corresponding Gaussian mixture model.
        """
        if g == 0:
            # one noise cluster case
            if not noise:
                raise ModelError("undefined model")
            return 1
        else:
            nparams = self.n_var_params(d, g) + g * d
            if not equalpro:
                nparams = nparams + (g - 1)
            if noise:
                nparams = nparams + 2
        return nparams


# TODO add vinv
class MixtureModel:
    """
    Abstract class representing a Gaussian finite mixture model.
    """
    def __init__(self, data, z=None, prior=None):
        """
        Gaussian finite mixture model.

        :param data: data to use for fitting the Gaussian finite mixture model represented by a numpy array, containing
        n observations, of dimension d.
        :param z: numpy array with shape (n x G), where n is the number of observations in data, and G the amount of
        cluster components in the model. index [i,j] indicates the probability that observation i belongs to component
        j.
        :param prior: optional prior for mixture model (not yet supported)
        """
        # Model type from Model enum.
        self.model = None
        self.data = data
        self.prior = prior

        self.mean = None
        self.variance = None
        # mixing proportions of model
        self.pro = None

        self.loglik = None
        self.returnCode = None
        self.z = z

        self.G = None
        self.n = len(data)
        if data.ndim == 1:
            self.d = 1
        else:
            self.d = data.shape[1]

    def bic(self, noise=False, equalpro=False):
        """
        Computes the BIC (Bayesian Information Criterion) for this mixture model.

        :param noise: A logical variable indicating whether or not the model includes an optional Poisson noise
        component. The default is to assume no noise component.
        :param equalpro: A logical variable indicating whether or not the components in the model are assumed to be
        present in equal proportion. The default is to assume unequal mixing proportions.
        :return: The BIC or Bayesian Information Criterion for the given input arguments.
        """
        if self.returnCode is None:
            raise ModelError("Model is not fitted yet, was fit called on this model?")
        nparams = self.model.n_mclust_params(self.d, self.G, noise, equalpro)
        return 2 * self.loglik - nparams * log(self.n)

    def fit(self):
        """
        Fits this mixture model on self.data.

        :return: returnCode of success of model.
        """
        raise AbstractMethodError()

    def classify(self):
        """
        Gives the allocation of observations in self.data to which cluster component.

        :return: numpy array containing index to which cluster component observation i belongs.
        """
        if self.returnCode is None:
            raise ModelError("Model is not fitted yet, was fit called on this model?")
        elif self.returnCode != 0:
            raise ModelError("Model not fitted correctly, check warnings and returnCode for more information.")
        if self.G == 1:
            return np.zeros(self.n)

    def component_density(self, new_data=None, logarithm=False):
        raise AbstractMethodError()

    def density(self, new_data=None, logarithm=False):
        if self.pro is None:
            raise ModelError("mixing proportions must be supplied")
        cden = self.component_density(new_data, logarithm=True)
        # TODO implement vinv/noise
        if self.G > 1:
            pro = np.copy(self.pro)
            if np.any(self.pro == 0):
                pro = pro[self.pro != 0]
                cden = cden[:, self.pro != 0]
            cden = cden + np.log(pro)

        maxlog = np.amax(cden, 1)
        cden = (cden.transpose() - maxlog).transpose()
        den = np.log(np.sum(np.exp(cden), 1)) + maxlog

        # TODO implement vinv/noise
        if not logarithm:
            den = np.exp(den)

        return den

    def __deepcopy__(self, memodict={}):
        new = type(self)(self.data, np.copy(self.z, order='F'))
        new.G = copy.deepcopy(self.G)
        new.mean = copy.deepcopy(self.mean)
        new.pro = copy.deepcopy(self.pro)

        new.variance = copy.deepcopy(self.variance)

        new.loglik = copy.deepcopy(self.loglik)
        new.returnCode = self.returnCode

        new.prior = self.prior

        return new

    def copy_onto(self, model):
        model.model = self.model
        model.data = self.data
        model.prior = self.prior

        model.mean = self.mean
        model.variance = self.variance
        model.pro = self.pro

        model.loglik = self.loglik
        model.returnCode = self.returnCode
        model.z = self.z

        model.G = self.G
        model.n = self.n
        model.d = self.d

    def __str__(self):
        return f"modelname: {self.model}\n" \
               f"n: {self.n}\n" \
               f"d: {self.d}\n" \
               f"G: {self.G}\n" \
               f"mean: {self.mean}\n" \
               f"variance: {self.variance}\n" \
               f"pro: {self.pro}\n" \
               f"loglik: {self.loglik}\n" \
               f"returnCode: {self.returnCode}\n" \
               f"prior: {self.prior}\n"

