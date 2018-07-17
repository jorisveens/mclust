from mclust.Models import Model
from mclust.MVN import *
from mclust.ME import *
from mclust.Utility import mclust_unmap, qclass
from mclust.hc import HCEII, HCVVV


class ModelFactory:
    def create(data, model, z=None, groups=2, prior=None):
        model = {
            Model.E: [MVNX, MEE],
            Model.V: [MVNX, MEV],
            Model.X: [MVNX, MEX],
            Model.EII: [MVNXII, MEEII],
            Model.VII: [MVNXII, MEVII],
            Model.EEI: [MVNXXI, MEEEI],
            Model.VEI: [MVNXXI, MEVEI],
            Model.EVI: [MVNXXI, MEEVI],
            Model.EEE: [MVNXXX, MEEEE],
            Model.VVV: [MVNXXX, MEVVV],
        }.get(model)
        if data.ndim == 1:
            if z is None:
                z = mclust_unmap(qclass(data, groups))
            if z.shape[1] == 1:
                return model[0](np.asfortranarray(data), z, prior)
            else:
                return model[1](np.asfortranarray(data), z, prior)
        else:
            if z is None:
                d = data.shape[1]
                if len(data) > d:
                    hc = HCVVV(data)
                else:
                    hc = HCEII(data)
                hc.fit()
                z = mclust_unmap(hc.get_class_matrix([groups])[:, 0])
            if z.shape[1] == 1:
                return model[0](np.asfortranarray(data), z, prior)
            else:
                return model[1](np.asfortranarray(data), z, prior)
    create = staticmethod(create)


