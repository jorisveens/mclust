from mclust.fortran import mclust
from mclust.Exceptions import *
import numpy as np
import sys
import warnings


class MVNData:

    def __init__(self, modelName, n, d, G, parameters, loglik, returnCode, prior=None):
        self.modelName = modelName
        self.prior = prior
        self.n = n
        self.d = d
        self.G = G
        self.parameters = parameters
        self.loglik = loglik
        self.ret = returnCode

    def __str__(self):
        return f"modelname: {self.modelName}\n" \
               f"n: {self.n}\n" \
               f"d: {self.d}\n" \
               f"G: {self.G}\n" \
               f"parameters: {self.parameters}\n" \
               f"loglik: {self.loglik}\n" \
               f"returnCode: {self.ret}\n" \
               f"prior: {self.prior}\n"


def mvnX(data, prior=None, warn=False):
    if data.ndim != 1:
        raise DimensionError("Data must be one dimensional.")

    floatdata = data.astype(float, order='F')
    ret = 0

    if prior is None:
        mu, sigmasq, loglik = mclust.mvn1d(floatdata)
    else:
        raise NotImplementedError()

    if loglik > sys.float_info.max:
        if warn:
            warnings.warn("sigma-squared vanishes")
        loglik = None
        ret = -1

    variance = {
        'modelName': 'X',
        'd': 1,
        'G': 1,
        'sigmasq': 1
    }

    parameters = {
        'pro': 1,
        'mean': mu,
        'variance': variance
    }

    return MVNData(
        modelName="X",
        prior=prior,
        n=len(data),
        d=1,
        G=1,
        parameters=parameters,
        loglik=loglik,
        returnCode=ret
    )


y = np.array([1,2,3,4,5,6,7,8])
