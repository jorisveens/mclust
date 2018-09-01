from mclust.em import *
from mclust.hierarchical_clustering import HCEII, HCVVV
from mclust.multi_var_normal import *
from mclust.utility import mclust_unmap, qclass


class ModelFactory:
    """
    This is a helper class, that can be used to set up model configurations.
    """

    @staticmethod
    def create(data, model, z=None, groups=2, prior=None, **kwargs):
        """
        Static method that selects and sets up a MixtureModel based on the given input.

        :param data: Data that is used for fitting the model. Represented by a NumPy array
                     with shape (n × d), where n is the number of observations, and d is
                     the dimension of the data.
        :param model: Mixture model configuration. Element of Model enumeration.
        :param z: Matrix with shape (n×g). The [i, k]th element indicates the conditional
                  probability of the observation i of data belonging to cluster component
                  k. This is an optional parameters, z or groups need to be supplied. If
                  both are supplied z is used as the initial guess for cluster assignments.
        :param groups: Integer specifying how many groups should be used in the resulting
                       mixture model. Hierarchical clustering is used to set up initial
                       guess for cluster assignments.
        :param prior: Not Implemented.
        :param kwargs: Optional parameters for model setup.
        :return:
        """
        model = {
            Model.E: [MVNX, MEE],
            Model.V: [MVNX, MEV],
            Model.X: [MVNX, MEE],
            Model.EII: [MVNXII, MEEII],
            Model.VII: [MVNXII, MEVII],
            Model.EEI: [MVNXXI, MEEEI],
            Model.VEI: [MVNXXI, MEVEI],
            Model.EVI: [MVNXXI, MEEVI],
            Model.VVI: [MVNXXI, MEVVI],
            Model.EEE: [MVNXXX, MEEEE],
            Model.EVE: [MVNXXX, MEEVE],
            Model.VEE: [MVNXXX, MEVEE],
            Model.VVE: [MVNXXX, MEVVE],
            Model.EEV: [MVNXXX, MEEEV],
            Model.VEV: [MVNXXX, MEVEV],
            Model.EVV: [MVNXXX, MEEVV],
            Model.VVV: [MVNXXX, MEVVV]
        }.get(model)
        if 'control' in kwargs:
            control = kwargs['control']
        else:
            control = EMControl()

        if data.ndim == 1:
            if z is None:
                z = mclust_unmap(qclass(data, groups))
            if z.shape[1] == 1:
                return model[0](np.asfortranarray(data), z, prior)
            else:
                return model[1](np.asfortranarray(data), z, prior, control)
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
                return model[1](np.asfortranarray(data), z, prior, control)


