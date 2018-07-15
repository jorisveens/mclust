from enum import Enum
from math import log

from mclust.Exceptions import ModelError, AbstractMethodError


class Model(Enum):
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

    X = 'X'
    XII = 'XII'
    XXI = 'XXI'
    XXX = 'XXX'

    def n_var_params(self, d, g):
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


class MixtureModel:
    def __init__(self, data, z=None, prior=None):
        self.model = None
        self.prior = None
        self.G = None
        self.mean = None
        self.pro = None

        self.variance = None

        self.loglik = None
        self.returnCode = None
        self.z = z

        self.data = data
        self.prior = prior
        self.n = len(data)
        if data.ndim == 1:
            self.d = 1
        else:
            self.d = data.shape[1]

    def bic(self, noise=False, equalpro=False):
        nparams = self.model.n_mclust_params(self.d, self.G, noise, equalpro)
        return 2 * self.loglik - nparams * log(self.n)

    def fit(self):
        raise AbstractMethodError()

    def classify(self):
        if self.returnCode is None:
            raise ModelError("Model is not fitted yet, was fit called on this model?")
        elif self.returnCode != 0:
            raise ModelError("Model not fitted correctly, check warnings and returnCode for more information.")

    def __str__(self):
        return f"modelname: {self.model}\n" \
               f"n: {self.n}\n" \
               f"d: {self.d}\n" \
               f"G: {self.G}\n" \
               f"mean: {self.mean}\n" \
               f"variance: {self.variance}\n" \
               f"pro: {self.pro}\n" \
               f"loglik: {self.loglik}\n" \
               f"returnCode: {self.returnCode}\n" \
               f"prior: {self.prior}\n"

