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
        self.scale = np.array(scale)
        if self.scale.ndim == 0:
            self.scale = np.repeat(scale, g)
        self.shape = np.array(shape)
        if self.shape.ndim == 1:
            self.shape = np.tile(shape, (self.G, 1))
        # TODO variant orientation
        if orientation is None:
            self.orientation = np.tile(np.identity(self.d), (self.G, 1)).reshape(self.G, self.d, self.d)
        elif orientation.ndim == 2:
            self.orientation = np.tile(orientation, (self.G, 1)).reshape(self.G, self.d, self.d)
        else:
            self.orientation = orientation

    def get_covariance(self):
        covariance = np.zeros((self.G, self.d, self.d))
        for i in range(self.G):
            covariance[i] = self.scale[i] * \
                            self.orientation[i].dot(np.diag(self.shape[i]))\
                                .dot(self.orientation[i].transpose())

        # return self.scale * self.orientation.dot(self.shape).dot(self.orientation.transpose())
        return covariance


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
