import warnings
from collections import defaultdict

import numpy as np

from mclust.control import EMControl, ModelTypes
from mclust.exceptions import ModelError
from mclust.hierarchical_clustering import HCEII, HCVVV
from mclust.model_factory import ModelFactory
from mclust.utility import qclass, mclust_unmap


class MclustBIC:
    def __init__(self, data, groups=None, models=None, prior=None, control=EMControl(), initialization={'noise': None}):
        """
        Object for fitting all provided (or default) model configurations on the specified range of cluster component
        numbers.

        :param data: The data that is used for fitting the model. Represented by a Fortran contiguous float NumPy array.
        :param groups: List of integers specifying the number of cluster components to fit all model configurations on.
        :param models: List of Model values, specifying the model configuration to fit.
        :param prior: Not yet implemented.
        :param control: EMControl object specifying the control parameters used to fit the models.
        :param initialization: Option dictionary, currently only noise=None is supported.
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
            # Use Hierarchical clustering to obtain starting classes
            hc_matrix = None
            if self.d != 1:
                if self.n > self.d:
                    hc = HCVVV(data)
                else:
                    hc = HCEII(data)
                hc.fit()
                hc_matrix = hc.get_class_matrix(self.groups)
            # fit all combinations of groups and model configurations
            for groupIndex, group in enumerate(self.groups):
                for modelIndex, model in enumerate(self.models):
                    if self.fitted_models[group, model] is not None:
                        continue

                    # use qclass to initialise one dimensional models else use hierarchical clustering
                    z = mclust_unmap(qclass(data, group)) if self.d == 1 else mclust_unmap(hc_matrix[:, groupIndex])
                    if min(np.apply_along_axis(sum, 0, z)) == 0:
                        warnings.warn("there are missing groups")

                    mod = ModelFactory.create(data, model, z=z, prior=prior, control=control)
                    mod.fit()
                    self.fitted_models[group, model] = mod

    def _handle_model_selection(self):
        """
        Sets default model configurations if none a applied
        """
        if self.models is None:
            if self.d == 1:
                self.models = ModelTypes.get_one_dimensional()
            else:
                self.models = ModelTypes.get_multi_dimensional()

    def _handle_group_selection(self, initialization):
        """
        Sets groups based on groups input and initialization

        If no groups are specified 1 to 9 groups are used by default
        :param initialization: Option dictionary
        """
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
                raise ModelError("g must be positive")
        else:
            if any([group < 0 for group in self.groups]):
                raise ModelError("g must be non-negative")

    def get_bic_matrix(self):
        """
        Calculate BIC value for all fitted models

        :return: NumPy array with shape (g × m) , where g is the length of groups and m is the length of models.
                 The [i, j]th element indicates the BIC value of configuration models[j] with groups[i] cluster
                 components.
        """
        bic_matrix = np.full((len(self.groups), len(self.models)), float('nan'), dtype=float)
        for group_index, group in enumerate(self.groups):
            for model_index, model in enumerate(self.models):
                fitted = self.fitted_models[group, model]
                if fitted is not None and fitted.return_code == 0:
                    bic_matrix[group_index, model_index] = fitted.bic()
        return bic_matrix

    def get_return_codes_matrix(self):
        """
        Collect the return value of all fitted models

        :return: NumPy array with shape (g × m) , where g is the length of groups and m is the length of models.
                 The [i, j]th element indicates the return code of configuration models[j] with groups[i] cluster
                 components.
        """
        ret_matrix = np.full((len(self.groups), len(self.models)), -42, dtype=int)
        for group_index, group in enumerate(self.groups):
            for model_index, model in enumerate(self.models):
                fitted = self.fitted_models[group, model]
                if fitted is not None:
                    ret_matrix[group_index, model_index] = fitted.return_code
        return ret_matrix

    def pick_best_model(self):
        """
        Selects the best fitted model based on BIC value.

        :return: MixtureModel corresponding to the best fitted model based on BIC.
        """
        bic_matrix = self.get_bic_matrix()
        try:
            index = np.unravel_index(np.nanargmax(bic_matrix), bic_matrix.shape)
        except ValueError:
            # No valid models
            return None
        return self.fitted_models[self.groups[index[0]], self.models[index[1]]]
