import numpy as np
from math import log
import warnings

from mclust.Exceptions import ModelError
from mclust.MVN import model_to_mvn
from mclust.ME import model_to_me
from mclust.Utility import qclass, unmap
from mclust.Models import Model


class BICData:
    def __init__(self, BICMatrix, returnCodes, groups, models):
        self.BICMatrix = BICMatrix
        self.returnCodes = returnCodes
        self.groups = groups
        self.models = models


def mclustBIC(data, g=None, models=None, prior=None, initialization={'noise': None}):
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
    data = data.astype(float, order='F')
    if data.ndim == 1:
        n = len(data)
        d = 1
    else:
        n, d = data.shape
    # TODO check if previous BIC values are available
    # TODO include noise
    if models is None:
        if d == 1:
            models = [Model.E, Model.V]
        else:
            # TODO fill out modelname selection
            models = [Model.EII, Model.VII]
    if g is None:
        # if no groups are specified generate groups with 1 to 9 elements
        g = list(range(1, 10))
    else:
        # only select unique number of elements
        g = [int(i) for i in list(set(g))]
        g.sort()
    if initialization['noise'] is None:
        g = [group for group in g if group <= n]
        if any([group <= 0 for group in g]):
            raise ModelError("G must be positive")
    else:
        if any([group < 0 for group in g]):
            raise ModelError("G must be non-negative")
    l = len(g)
    m = len(models)
    bic_values = np.full((l, m), None)
    return_codes = np.full((l, m), None)

    if initialization['noise'] is None:
        # FIXME possible merge with other groups be using model_to_x(model, g)
        if g[0] == 1:
            for modelIndex, model in enumerate(models):
                if bic_values[0, modelIndex] is None:
                    mod = model_to_mvn(model)
                    ret_code = mod.fit(data, prior)
                    return_codes[0, modelIndex] = ret_code
                    bic_values[0, modelIndex] = bic(mod, equalpro=False)
        # TODO pre specified hcpairs
        if d != 1:
            pass
            # if (n > d):
            #     hcPairs = hc(data=data,
            #                    modelName=mclust.options("hcModelNames")[1])
            # else:
            #     hcPairs = hc(data = data, modelName = "EII")
        else:
            hcPairs = None
        #   hcPairs <- hc(data = data, modelName = "E")
        # FIXME for now only case for 1 dimension
        for groupIndex, group in enumerate(g):
            if group == 1:
                continue
            for modelIndex, model in enumerate(models):

                if bic_values[groupIndex, modelIndex] is not None:
                    continue

                # TODO replace random z with hierarchichal clustering
                z = unmap(qclass(data, group)) if d == 1 else random_z(n, group)
                if min(np.apply_along_axis(sum, 0, z)) == 0:
                    warnings.warn("there are missing groups")

                # FIXME pass control parameter
                mod = model_to_me(model)
                ret_code = mod.fit(data, z, prior)
                bic_values[groupIndex, modelIndex] = bic(mod, equalpro=False)
                return_codes[groupIndex, modelIndex] = ret_code

    return BICData(bic_values, return_codes, g, models)


def bic(fitted_model, noise=False, equalpro=False):
    nparams = fitted_model.model.n_mclust_params(fitted_model.d, fitted_model.G, noise, equalpro)
    return 2 * fitted_model.loglik - nparams * log(fitted_model.n)


def random_z(n, g):
    z = np.zeros((n, g), float, order='F')
    for i in range(n):
        sum = 1.0
        for j in range(g-1):
            rand = np.random.uniform(high=sum)
            z[i, j] = rand
            sum -= rand
        z[i, g-1] = sum
    return z





