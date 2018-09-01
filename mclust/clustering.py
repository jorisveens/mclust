import warnings

from mclust.bic import MclustBIC
from mclust.control import EMControl
from mclust.models import Model, MixtureModel


class Mclust(MixtureModel):
    """
    main entry point for the clustering functionality of mclust. It fits all
    provided or default models on a specified number of cluster components. After calling the
    fit() function, Mclust objects behave like the best model, based on BIC value.
    """
    def __init__(self, data, groups=None, models=None, prior=None, control=EMControl()):
        """
        Constructor

        :param data: Data that is used for fitting the model. Represented by a NumPy array
                     with shape (n × d), where n is the number of observations, and d is the
                     dimension of the data.
        :param groups: List containing the number of cluster components that should be
                       considered for the different models and comparison. By default 1 to 9
                       cluster components are considered.
        :param models: List of model configurations used for fitting and comparison. By default
                       all possible models are considered.
        :param prior: Not implemented.
        :param control: EMControl object, that is used to specify tolerance for convergence,
                        iteration limit and whether the mixing proportions should be assumed
                        to be equal.
        """
        super().__init__(data, prior)
        self.groups = groups
        self.models = models
        self.control = control
        self._underlying_model = None

    def fit(self):
        bic = MclustBIC(self.data, self.groups, self.models, self.prior)
        self.groups = bic.groups
        self.models = bic.models
        model = bic.pick_best_model()
        if model is None:
            return None
        if model.g == max(self.groups):
            warnings.warn("optimal number of clusters occurs at max choice")
        elif model.g == min(self.groups):
            warnings.warn("optimal number of clusters occurs at min choice")

        self.__dict__.update(model.__dict__.copy())
        self._underlying_model = model
        return self.return_code

    def classify(self):
        super().classify()
        return self._underlying_model.classify()

    def component_density(self, new_data=None, logarithm=False):
        return self._underlying_model.component_density(new_data, logarithm)

