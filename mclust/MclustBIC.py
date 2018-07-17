import numpy as np
from collections import defaultdict
import warnings

from mclust.Exceptions import ModelError
from mclust.Utility import qclass, mclust_unmap
from mclust.Models import Model
from mclust.hc import HCEII, HCVVV
from mclust.modelfactory import ModelFactory


class MclustBIC:
    def __init__(self, data, groups=None, models=None, prior=None, initialization={'noise': None}):
        """
        Calculate BIC values for data for g groups models specified  by modelNames.

        :raises ModelError if G or modelNames are incorrectly specified.
        :modifies g, modelnames

        :param data: data used for fitting the models
        :param g: list of amount of groups to consider, if none are applied use 1 to 9 groups
        :param models: list of models that are fitted, if this is None all available models will be fitted
        :param prior: prior belief of shape direction and scale
        :param initialization: initialization parameters
        :return: BICData containing BIC values and return codes for g groups and models specified by modelnames
        """

        self.groups = groups
        self.models = models
        data = data.astype(float, order='F')
        if data.ndim == 1:
            self.n = len(data)
            self.d = 1
        else:
            self.n, self.d = data.shape

        self._handle_model_selection()
        self._handle_group_selection(initialization)
        self.fitted_models = defaultdict(lambda: None)

        if initialization['noise'] is None:
            # possibly merge with other groups be using model_to_x(model, g)
            if self.groups[0] == 1:
                for modelIndex, model in enumerate(self.models):
                    if self.fitted_models[model, 1] is None:
                        mod = ModelFactory.create(data, model, groups=1, prior=prior)
                        mod.fit()
                        self.fitted_models[model, 1] = mod

            # Use Hierarchical clustering to obtain starting classes
            hc_matrix = None
            if self.d != 1:
                if self.n > self.d:
                    hc = HCVVV(data)
                else:
                    hc = HCEII(data)
                hc.fit()
                hc_matrix = hc.get_class_matrix(self.groups)
            for groupIndex, group in enumerate(self.groups):
                if group == 1:
                    continue
                for modelIndex, model in enumerate(self.models):
                    if self.fitted_models[model, group] is not None:
                        continue

                    z = mclust_unmap(qclass(data, group)) if self.d == 1 else mclust_unmap(hc_matrix[:, groupIndex])
                    if min(np.apply_along_axis(sum, 0, z)) == 0:
                        warnings.warn("there are missing groups")

                    # FIXME pass control parameter
                    mod = ModelFactory.create(data, model, z=z, prior=prior)
                    mod.fit()
                    self.fitted_models[model, group] = mod

    def _handle_model_selection(self):
        if self.models is None:
            if self.d == 1:
                self.models = [Model.E, Model.V]
            else:
                # TODO fill out modelname selection
                self.models = [Model.EII, Model.VII, Model.EEI, Model.VEI,
                               Model.EVI, Model.EEE, Model.VVV]

    def _handle_group_selection(self, initialization):
        if self.groups is None:
            # if no groups are specified generate groups with 1 to 9 elements
            self.groups = list(range(1, 10))
        else:
            # only select unique number of elements
            self.groups = [int(i) for i in list(set(self.groups))]
            self.groups.sort()
        if initialization['noise'] is None:
            self.groups = [group for group in self.groups if group <= self.n]
            if any([group <= 0 for group in self.groups]):
                raise ModelError("G must be positive")
        else:
            if any([group < 0 for group in self.groups]):
                raise ModelError("G must be non-negative")

    def get_bic_matrix(self):
        bic_matrix = np.full((len(self.groups), len(self.models)), None)
        for group_index, group in enumerate(self.groups):
            for model_index, model in enumerate(self.models):
                fitted = self.fitted_models[model, group]
                if fitted is not None:
                    bic_matrix[group_index, model_index] = fitted.bic()
        return bic_matrix

    def get_return_codes_matrix(self):
        ret_matrix = np.full((len(self.groups), len(self.models)), None)
        for group_index, group in enumerate(self.groups):
            for model_index, model in enumerate(self.models):
                fitted = self.fitted_models[model, group]
                if fitted is not None:
                    ret_matrix[group_index, model_index] = fitted.returnCode
        return ret_matrix

    def pick_best_model(self):
        bic_matrix = self.get_bic_matrix()
        index = np.unravel_index(np.argmax(bic_matrix), bic_matrix.shape)
        return self.fitted_models[self.models[index[1]], self.groups[index[0]]]
