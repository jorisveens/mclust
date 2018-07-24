import warnings

from mclust.control import EMControl
from mclust.mclust_bic import MclustBIC

from mclust.models import Model, MixtureModel


class Mclust(MixtureModel):
    def __init__(self, data, groups=None, models=None, prior=None, control=EMControl()):
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
        if model.G == max(self.groups):
            warnings.warn("optimal number of clusters occurs at max choice")
        elif model.G == min(self.groups):
            warnings.warn("optimal number of clusters occurs at min choice")

        self.__dict__.update(model.__dict__.copy())
        self._underlying_model = model

    def classify(self):
        super().classify()
        return self._underlying_model.classify()
