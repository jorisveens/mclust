import numpy as np

from mclust.Exceptions import ModelError
from mclust.hc import HCVVV, HCEII
from mclust.Utility import mclust_unmap, qclass
from mclust.MVN import MVNX, MVNXXX
from mclust.ME import MEE, MEVVV


class ModelE:
    def __new__(cls, data, group=2, prior=None):
        if data.ndim != 1:
            raise ModelError("Data should be 1 dimensional for E model")
        z = mclust_unmap(qclass(data, group))
        if group == 1:
            model = MVNX
        else:
            model = MEE
        return model(np.asfortranarray(data), z, prior)

    def __init__(self, data, group, prior=None):
        pass


class ModelVVV:
    def __new__(cls, data, group, prior=None):
        if data.ndim != 2:
            raise ModelError("Data should be 2 dimensional for VVV model")
        d = data.shape[1]
        if len(data) > d:
            hc = HCVVV(data)
        else:
            hc = HCEII(data)
        hc.fit()
        z = mclust_unmap(hc.get_class_matrix([group])[:, 0])
        if group == 1:
            model = MVNXXX
        else:
            model = MEVVV
        return model(np.asfortranarray(data), z, prior)

    def __init__(self, data, group, prior=None):
        pass
