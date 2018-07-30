import numpy as np

from mclust.exceptions import ModelError
from mclust.control import EMControl, ModelTypes


class MclustDA:
    def __init__(self, data, classes, g=None, models=None, model_type=None, control=EMControl()):
        if classes is None:
            raise ValueError("class labels (classes) for training data must be provided")
        self._handle_input(data, classes, g, models)

        # continue implement MclustDA

    def _handle_input(self, data, classes, g, models):
        self.data = data
        self.n, self.p = data.shape
        self.labels = list(set(classes))

        if len(self.labels) == 1:
            self.g = 1

        # what is the use of a list here
        self.g = {}
        if g is None:
            for i in range(len(self.labels)):
                self.g[i] = np.arange(1, 6)
        elif isinstance(g, dict):
            for key in g.keys():
                self.g[key] = np.sort(g[key])
        else:
            for i in range(len(self.labels)):
                self.g[i] = np.sort(g)

        if np.any([np.any(vals <= 0) for vals in self.g.values()]):
            raise ModelError("all values of G must be positive")

        if models is None:
            if self.p == 1:
                models = ModelTypes.get_one_dimensional()
            else:
                models = ModelTypes.get_multi_dimensional()
        if self.n <= self.p:
            models = list(set(ModelTypes.get_multi_dimensional()).intersection(ModelTypes.get_less_observations()))

        if not isinstance(models, dict):
            self.models = {}
            for i in range(len(self.labels)):
                self.models[i] = models



