import copy
from enum import Enum

import numpy as np

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

    :field model: Model configuration, element of Model enumeration
    :field data: Data that is used for fitting the model. Represented by a NumPy
                 array with shape (n × d), where n is the number of observations, and
                 d is the dimension of the data.
    :field g: Number of cluster components in the mixture model.
    :field n: Number of observations in data.
    :field d: Dimension of observation in data.
    :field mean: Means of the cluster components. Represented by a NumPy array
                 with shape (g × d). The ith component of mean is the cluster centre
                 of group i.
    :field variance: Covariance matrices of the cluster components. Represented by a
                     NumPy array with shape (g × d × d) The ith component represents
                     the covariance matrix of group i.
    :field pro: Mixing proportions (τ ) of the mixture model. Represented by a
                NumPy array with shape g.
    :field loglik: Maximised Log-likelihood of data in fitted mixture model.
    :field z: Matrix of shape (n × d), the [i, k]th component indicates the condi-
              tional probability that the ith observation belongs to cluster compo-
              nent k.
    :field return_code: return code of the last method used on the model (fit, e_step or m_step). Possible return codes:
                       0 : method completed successfully.
                       -1: One or more cluster components have a singular covariance.
                       -2: Mixing proportion fell below minimum threshold, with
                       equal mixing proportions.
                       -3: Mixing proportion fell below minimum threshold.
                       -9: Internal error in model fitting.
                       1 : Iteration limit reached.
                       9 : mean, variance, pro or z is missing, or contains missing values
    """
    def __init__(self, data, z=None, prior=None, **kwargs):
        """
        Gaussian finite mixture model.

        :param data: data to use for fitting the Gaussian finite mixture model represented by a numpy array, containing
                     n observations, of dimension d.
        :param z: numpy array with shape (n x g), where n is the number of observations in data, and G the amount of
                  cluster components in the models. index [i,j] indicates the probability that observation i belongs to
                  component j.
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
        self.return_code = None
        self.z = z

        self.g = None
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
        if self.return_code is None:
            raise ModelError("Model is not fitted yet, was fit called on this model?")
        nparams = self.model.n_mclust_params(self.d, self.g, noise, equalpro)
        return 2 * self.loglik - nparams * np.log(self.n)

    def fit(self):
        """
        Fits this mixture model on self.data.

        :return: return_code of success of model.
        """
        raise AbstractMethodError()

    def classify(self):
        """
        Gives the allocation of observations in self.data to which cluster component.

        :return: numpy array containing index to which cluster component observation i belongs.
        """
        if self.return_code is None:
            raise ModelError("Model is not fitted yet, was fit called on this model?")
        elif self.return_code != 0:
            raise ModelError("Model not fitted correctly, check warnings and return_code for more information.")
        if self.g == 1:
            return np.zeros(self.n)

    def predict(self, new_data=None):
        """
        Computes the cluster predictions for new_data.

        If new_data is not provided the cluster predictions of the training data are computed
        :param new_data: An optional matrix containing data observations of dimension d. If
                         no data is specified, the cluster assignment for the training data are
                         returned.
        :return: Vector containing predicted cluster assignments for new_data. Indices
                 of return vector correspond to indices in new_data.
        """
        if new_data is None:
            new_data = self.data
        else:
            if new_data.ndim == 1:
                if self.d != 1:
                    raise ModelError("new_data does not have the same dimensions as training data")
            elif new_data.shape[1] != self.d:
                raise ModelError("new_data does not have the same dimensions as training data")

        z = self.component_density(new_data, logarithm=True)
        # TODO implement vinv/noise
        z = z + np.log(self.pro / sum(self.pro))
        z = (z.transpose() - np.log(np.sum(np.exp(z), 1))).transpose()
        z = np.exp(z)
        return np.argmax(z, 1)

    def component_density(self, new_data=None, logarithm=False):
        """
        Calculates the component densities for observations in this mixture model.

        :param new_data: An optional matrix with containing data observations of dimension
                         d. If no data is specified, the component densities for the training
                         data are calculated.
        :param logarithm: Boolean indicating whether or not the logarithm of the component
                          densities should be returned
        :return: Matrix containing component densities of new_data. The [i, k]th
                 component indicates the density of observation i in component k.
        """
        raise AbstractMethodError()

    def density(self, new_data=None, logarithm=False):
        """
        Calculates the densities of observations in this mixture model.

        :param new_data: An optional matrix with containing data observations of dimension
                         d. If no data is specified, the densities for the training data are
                         calculated.
        :param logarithm: Boolean indicating whether or not the logarithm of the densities
                          should be returned.
        :return: Vector containing densities of new_data. The ith component indicates the density
                 of observation i in the mixture model.
        """
        if self.pro is None:
            raise ModelError("mixing proportions must be supplied")
        cden = self.component_density(new_data, logarithm=True)
        # TODO implement vinv/noise
        if self.g > 1:
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
        new.g = copy.deepcopy(self.g)
        new.mean = copy.deepcopy(self.mean)
        new.pro = copy.deepcopy(self.pro)

        new.variance = copy.deepcopy(self.variance)

        new.loglik = copy.deepcopy(self.loglik)
        new.return_code = self.return_code

        new.prior = self.prior

        return new

    def copy_onto(self, model):
        """
        Copies the field of this model (self) onto model.

        :param model: MixutureModel to copy field to
        """
        model.model = self.model
        model.data = self.data
        model.prior = self.prior

        model.mean = self.mean
        model.variance = self.variance
        model.pro = self.pro

        model.loglik = self.loglik
        model.return_code = self.return_code
        model.z = self.z

        model.g = self.g
        model.n = self.n
        model.d = self.d

    def __str__(self):
        return "modelname: {}\n"\
               "n: {}\n" \
               "d: {}\n" \
               "g: {}\n" \
               "mean:\n{}\n" \
               "variance:\n{}\n" \
               "pro: {}\n" \
               "loglik: {}\n" \
               "return_code: {}\n".format(self.model, self.n, self.d, self.g, self.mean,
                                         self.variance, self.pro, self.loglik, self.return_code)

