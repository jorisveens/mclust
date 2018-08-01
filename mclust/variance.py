import numpy as np

from mclust.exceptions import DimensionError


class Variance:
    def __init__(self, d, g):
        self.d = d
        self.G = g

    def get_covariance(self):
        pass

    def select_group(self, index):
        self.G = 1

    def __str__(self):
        return f"covariance list:\n {self.get_covariance()}"


class VarianceSigmasq(Variance):
    def __init__(self, d, g, sigmasq):
        super().__init__(d, g)
        if sigmasq.ndim == 0:
            self.sigmasq = np.repeat(sigmasq, g)
        else:
            self.sigmasq = np.array(sigmasq)

    def get_covariance(self):
        covariance = np.zeros((self.G, self.d, self.d))
        for i, sq in enumerate(self.sigmasq):
            covariance[i] = np.identity(self.d) * sq
        return covariance

    def select_group(self, index):
        super().select_group(index)
        self.sigmasq = np.repeat(self.sigmasq[index], self.G)


class VarianceDecomposition(Variance):
    def __init__(self, d, g, scale, shape, orientation=None):
        super().__init__(d, g)
        self.scale = np.array(scale)
        if self.scale.ndim == 0:
            self.scale = np.repeat(scale, g)
        self.shape = np.array(shape)
        if self.shape.ndim == 1:
            self.shape = np.tile(shape, (self.G, 1))
        if orientation is None:
            self.orientation = np.tile(np.identity(self.d), (self.G, 1, 1))
        elif orientation.ndim == 2:
            self.orientation = np.tile(orientation, (self.G, 1, 1))
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

    def select_group(self, index):
        super().select_group(index)
        self.scale = np.repeat(self.scale[index], self.G)
        self.shape = np.tile(self.shape[index], (self.G, 1))
        self.orientation = np.tile(self.orientation[index], (self.G, 1, 1))


class VarianceCholesky(Variance):
    def __init__(self, d, g, cholsigma):
        super().__init__(d, g)
        if cholsigma.ndim != 3:
            raise DimensionError("cholsigma should have 3 dimensions")
        if cholsigma.shape[0] == 1:
            self.cholsigma = np.tile(cholsigma[0], (self.G, 1, 1))
        else:
            self.cholsigma = cholsigma

    def get_covariance(self):
        result = np.zeros((self.G, self.d, self.d))
        for g in range(self.G):
            result[g] = self.cholsigma[g].transpose().dot(self.cholsigma[g])
        return result

    def select_group(self, index):
        super().select_group(index)
        self.cholsigma = np.tile(self.cholsigma[index], (self.G, 1, 1))
