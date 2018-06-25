import numpy as np

from mclust.Exceptions import DimensionError


class Variance:
    def __init__(self, d, g):
        self.d = d
        self.G = g

    def get_covariance(self):
        pass

    def __str__(self):
        return f"covariance list:\n {self.get_covariance()}"


class VarianceSigmasq(Variance):
    def __init__(self, d, g, sigmasq):
        super().__init__(d, g)
        self.sigmasq = np.array(sigmasq)

    def get_covariance(self):
        if self.sigmasq.ndim == 0:
            # Model E and EII
            return np.tile(self.sigmasq * np.identity(self.d), [self.G, 1]).reshape(self.G, self.d, self.d)
        else:
            # Model V en VII
            covariance = np.zeros((self.G, self.d, self.d))
            for i in range(len(self.sigmasq)):
                covariance[i] = np.identity(self.d) * self.sigmasq[i]
            return covariance


# TODO make sure covariance returs a list of dimension g, d, d
class VarianceDecomposition(Variance):
    def __init__(self, d, g, scale, shape, orientation=None):
        super().__init__(d, g)
        self.scale = scale
        self.shape = shape
        self.orientation = np.identity(d) if orientation is None else orientation

    def get_covariance(self):
        return self.scale * self.orientation.dot(self.shape).dot(self.orientation.transpose())


class VarianceCholesky(Variance):
    def __init__(self, d, g, cholsigma):
        super().__init__(d, g)
        if cholsigma.ndim != 3:
            raise DimensionError("cholsigma should have 3 dimensions")
        self.cholsigma = cholsigma

    def get_covariance(self):
        # TODO  implement unchol
        result = np.zeros((self.G, self.d, self.d))
        if self.cholsigma.shape[0] == 1:
            for g in range(self.G):
                result[g] = self.cholsigma[0].transpose().dot(self.cholsigma[0])
        else:
            for g in range(self.G):
                result[g] = self.cholsigma[g].transpose().dot(self.cholsigma[g])
        return result
