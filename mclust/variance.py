import numpy as np

from mclust.Exceptions import DimensionError


# TODO make sure covariance returs a list of dimension g, d, d
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
        self.sigmasq = sigmasq

    def get_covariance(self):
        return self.sigmasq * np.identity(self.d)


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
        print(self.get_covariance())

    def get_covariance(self):
        # TODO  implement unchol
        result = self.cholsigma.copy()
        for g in range(self.G):
            result[g] = result[g].transpose().dot(result[g])
        return result
