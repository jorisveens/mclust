import copy
import warnings

import numpy as np

from mclust.bic import MclustBIC
from mclust.control import EMControl, ModelTypes
from mclust.exceptions import ModelError, AbstractMethodError
from mclust.model_factory import ModelFactory
from mclust.utility import mclust_unmap


class DiscriminantAnalysis:
    """
    Abstract class representing a classifier based on model-based clustering.
    """
    def __init__(self, data, labels, g=None, models=None, control=EMControl()):
        """
        Abstract class representing a classifier based on model-based clustering.

        :param data: The training data that is used for fitting the classifier. Represented by a Fortran contiguous
                     float NumPy array with shape (n x d) where n is the number of observations, and d the dimension of
                     the data.
        :param labels: Single dimensional NumPy integer array, representing the class labeling of data.
        :param g: Dictionary containing the amount of clusters used to represent each class, indexed by labels
        :param models: List of model configurations that should be considered by the classifier.
        :param control: EMControl object specifying the control parameters used to fit the models.
        """
        if labels is None:
            raise ValueError("class labels (labels) for training data must be provided")
        self.data = data
        n = data.shape[0]
        if data.ndim == 1:
            self.d = 1
        else:
            self.d = data.shape[1]
        self.labels = list(set(labels))
        self.nclasses = len(self.labels)
        self.fitted_models = {}
        self.observations = {}
        self.g = {}
        self.n = {}

        # set up amount of groups to consider
        if g is None:
            for i in range(self.nclasses):
                self.g[i] = np.arange(1, 6)
        elif isinstance(g, dict):
            for key in g.keys():
                self.g[key] = np.sort(g[key])
        else:
            for i in range(self.nclasses):
                self.g[i] = np.sort(g)

        if np.any([np.any(vals <= 0) for vals in self.g.values()]):
            raise ModelError("all values of g must be positive")

        # set up model configurations to consider
        if models is None:
            if self.d == 1:
                models = ModelTypes.get_one_dimensional()
            else:
                models = ModelTypes.get_multi_dimensional()
        if n <= self.d:
            models = list(set(ModelTypes.get_multi_dimensional()).intersection(ModelTypes.get_less_observations()))

        self.models = models
        if not isinstance(models, dict):
            self.models = {}
            for i in range(len(self.labels)):
                self.models[i] = models

    def predict(self, new_data=None, prior=None):
        """
        Computes classification prediction of new_data.

        If new_data is not specified the classification prediction of the training data is computed.

        :param new_data: NumPy array containing data points with dimension d (same as data). If new_data is None
                         the training data is used.
        :param prior: Optional NumPy float vector of length nclasses, containing the prior probabilities for each
                      class.
        :return: Single dimensional NumPy array containing predicted class assignments for new_data.
                 Indices of the return vector correspond with the indices of new_data.
        """
        group_sizes = np.array([model.n for model in self.fitted_models.values()])
        if new_data is None:
            new_data = self.data

        if prior is None:
            prior = group_sizes / np.sum(group_sizes)
        else:
            if len(prior) != self.nclasses:
                raise ModelError("wrong number of prior probabilities")
            if np.any(prior < 0):
                raise ModelError("prior must be nonnegative")

        z = np.zeros((new_data.shape[0], self.nclasses))
        for key in range(self.nclasses):
            z[:, key] = self.fitted_models[key].density(new_data, logarithm=True)
        z = z + np.log(prior/np.sum(prior))
        z = (z.transpose() - np.log(np.sum(np.exp(z), 1))).transpose()
        z = np.exp(z)
        cl = np.argmax(z, 1)

        return cl

    def df(self):
        """
        Computes the number of estimated parameters in the model used for classification.

        :return: Number of estimated parameters in the model used for classification.
        """
        raise AbstractMethodError()

    def loglik(self, new_data=None):
        """
        Computes the maximised log-likelihood of the new_data in model that is used by the classifier.

        :param new_data: NumPy array containing data points with dimension d (same as data). If new_data is None
                         the training data is used.
        :return:
        """
        if new_data is None:
            new_data = self.data
        n = np.sum(list(self.n.values()))
        logfclass = [np.log(self.n[key]/n) for key in sorted(self.n.keys())]
        ll = np.array([mod.density(new_data, True) for mod in self.fitted_models.values()])
        return np.sum(np.log(np.sum(np.exp(ll.transpose() + logfclass), 1)))


class EDDA(DiscriminantAnalysis):
    """
    Represents a classifier based on the EDDA method.
    """
    def __init__(self, data, labels, g=None, models=None, control=EMControl()):
        """
        Class representing a classifier based on the EDDA method.

        :param data: The training data that is used for fitting the classifier. Represented by a Fortran contiguous
                     float NumPy array with shape (n x d) where n is the number of observations, and d the dimension of
                     the data.
        :param labels: Single dimensional NumPy integer array, representing the class labeling of data.
        :param g: Dictionary containing the amount of clusters used to represent each class, indexed by labels
        :param models: List of model configurations that should be considered by the classifier.
        :param control: EMControl object specifying the control parameters used to fit the models.
        """
        super().__init__(data, labels, g, models, control)

        z = mclust_unmap(labels)

        best_model = None
        for model in self.models[0]:
            mod = ModelFactory.create(self.data, model, z.copy(order='F'), control=control)
            mod.m_step()
            mod.e_step()
            bic = mod.bic()
            if not np.isnan(bic) and (best_model is None or bic >= best_model.bic()):
                best_model = mod
        if best_model is None:
            warnings.warn("No model(s) can be estimated!")
            return

        for l in range(self.nclasses):
            ind = labels == self.labels[l]
            self.fitted_models[l] = copy.deepcopy(best_model)
            self.fitted_models[l].n = np.sum(ind)
            self.fitted_models[l].g = 1
            self.fitted_models[l].pro = np.array([1])
            self.fitted_models[l].mean = np.array([self.fitted_models[l].mean[l]])
            self.fitted_models[l].variance.select_group(l)
            self.fitted_models[l].z = None

            self.observations[l] = np.where(ind)
            self.n[l] = np.sum(ind)
            self.g[l] = 1

    def df(self):
        return int(self.d * self.nclasses + self.fitted_models[0].model.n_var_params(self.d, self.nclasses))


class MclustDA(DiscriminantAnalysis):
    """
    Represents a classifier based on the MclustDA method.
    """
    def __init__(self, data, labels, g=None, models=None, control=EMControl()):
        """
        Class representing a classifier based on the MclustDA method.

        :param data: The training data that is used for fitting the classifier. Represented by a Fortran contiguous
                     float NumPy array with shape (n x d) where n is the number of observations, and d the dimension of
                     the data.
        :param labels: Single dimensional NumPy integer array, representing the class labeling of data.
        :param g: Dictionary containing the amount of clusters used to represent each class, indexed by labels
        :param models: List of model configurations that should be considered by the classifier.
        :param control: EMControl object specifying the control parameters used to fit the models.
        """
        super().__init__(data, labels, g, models, control)
        for l in range(self.nclasses):
            ind = labels == self.labels[l]
            bic = MclustBIC(self.data[ind], self.g[l], self.models[l], control=control)
            self.fitted_models[l] = bic.pick_best_model()

            self.observations[l] = np.where(ind)
            self.g[l] = self.fitted_models[l].g
            self.n[l] = np.sum(ind)

    def df(self):
        return int(np.sum([(mod.g -1) + mod.g * self.d + mod.model.n_var_params(self.d, mod.g)
                           for mod in self.fitted_models.values()]))
