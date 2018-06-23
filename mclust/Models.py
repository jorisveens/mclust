from enum import Enum

from mclust.Exceptions import ModelError


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

    def __init__(self):
        self.model = None
        self.prior = None
        self.n = None
        self.d = None
        self.G = None
        self.mean = None
        self.pro = None

        self.variance = None

        self.loglik = None
        self.returnCode = None

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

# def nMclustParams(modelName, d, G, noise=False, equalpro=False):
#     nameSwitch = {
#         'X': "E",
#         'XII': "EII",
#         'XXI': "EEI",
#         'XXX': "EEE",
#     }
#     modelName = nameSwitch.get(modelName, modelName)
#     if G == 0:
#         # one noise cluster case
#         if not noise:
#             raise ModelError("undefined model")
#         nparams = 1
#     else:
#         nparams = nVarParams(modelName, d=d, G=G) + G * d
#         if not equalpro:
#             nparams = nparams + (G - 1)
#         if noise:
#             nparams = nparams + 2
#     return nparams
#
#
# def nVarParams(modelName, d, G):
#     model_switch = {
#         "E": 1,
#         "V": G,
#         "EII": 1,
#         "VII": G,
#         "EEI": d,
#         "VEI": G + (d - 1),
#         "EVI": 1 + G * (d - 1),
#         "VVI": G * d,
#         "EEE": d * (d + 1) / 2,
#         "EVE": 1 + G * (d - 1) + d * (d - 1) / 2,
#         "VEE": G + (d - 1) + d * (d - 1) / 2,
#         "VVE": G + G * (d - 1) + d * (d - 1) / 2,
#         "EEV": 1 + (d - 1) + G * d * (d - 1) / 2,
#         "VEV": G + (d - 1) + G * d * (d - 1) / 2,
#         "EVV": 1 - G + G * d * (d + 1) / 2,
#         "VVV": G * d * (d + 1) / 2,
#     }
#     result = model_switch.get(modelName, None)
#     if result is None:
#         raise ModelError("Invalid Model Name.")
#     return result
