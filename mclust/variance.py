import numpy as np

from mclust.exceptions import DimensionError


class Variance:
    """
    Representation for Covariance structures of a MixtureModel
    """
    def __init__(self, d, g):
        """
        Constructor

        :param d: Dimension of the data in MixtureModel
        :param g: Number of cluster components in MixtureModel
        """
        self.d = d
        self.g = g

    def get_covariance(self):
        """
        Computes the covariance matrices of the cluster components. Represented
        by a NumPy array with shape (g × d × d). The ith component represents
        the covariance matrix of group i.
        :return: NumPy array with shape (g × d × d). The ith component represents
                 the covariance matrix of group i.
        """
        pass

    def select_group(self, index):
        """
        Abstract method to transform the Variance to a single covariance matrix
        indicated by index.
        :param index: Index of covariance matrix to select
        """
        self.g = 1

    def __str__(self):
        return str(self.get_covariance())


class VarianceSigmasq(Variance):
    """
    Representation for Covariance structures of a MixtureModel based on vector
    of sigmasq (lambda) indicating the volume (scale) of the covariance.

    Used for E, V, EII and VII model configurations.
    """
    def __init__(self, d, g, sigmasq):
        """
        Constructor

        :param d: Dimension of the data in MixtureModel
        :param g: Number of cluster components in MixtureModel
        :param sigmasq: NumPy array of dimension 0 or 1, representing the scale of the covariance.
                      If scale is a scalar the scale is taken to be equal for all cluster components.
                      Otherwise the ith component of scale indicates the scale of the cluster component i.
        """
        super().__init__(d, g)
        if sigmasq.ndim == 0:
            self.sigmasq = np.repeat(sigmasq, g)
        else:
            self.sigmasq = np.array(sigmasq)

    def get_covariance(self):
        covariance = np.zeros((self.g, self.d, self.d))
        for i, sq in enumerate(self.sigmasq):
            covariance[i] = np.identity(self.d) * sq
        return covariance

    def select_group(self, index):
        super().select_group(index)
        self.sigmasq = np.repeat(self.sigmasq[index], self.g)


class VarianceDecomposition(Variance):
    """
    Representation for Covariance structures of a MixtureModel based on
    eigenvalue decomposition (scale, shape, orientation)

    Used for EEI, VEI, EVI, VVI, VEE, EVE, VVE, EEV, VEV and EVV, model
    configurations.
    """
    def __init__(self, d, g, scale, shape, orientation=None):
        """
        Constructor

        :param d: Dimension of the data in MixtureModel
        :param g: Number of cluster components in MixtureModel
        :param scale: NumPy array of dimension 0 or 1, representing the scale of the covariance.
                      If scale is a scalar the scale is taken to be equal for all cluster components.
                      Otherwise the ith component of scale indicates the scale of the cluster component i.
        :param shape: Optional NumPy array of dimension 1 or 2, representing the shape of the covariance.
                      where the ith row indicates the shape of the ith component, if shape has 2 dimensions.
                      If shape is one dimensional the shape is taken to be equal for all cluster components.
                      If no shape is supplied all cluster components are assumed to be spherical.
        :param orientation: Optional NumPy array of dimension 2 or 3, representing the orientation of the covariance.
                            The ith index represents the orientation matrix of cluster component i, when
                            orientation is 3 dimensional. If orientation is 2 dimensional, the orientation
                            for each cluster component is taken to be identical. If orientation is not supplied,
                            the identity matrix is taken as orientation matrix for all cluster components.
        """
        super().__init__(d, g)
        self.scale = np.array(scale)
        if self.scale.ndim == 0:
            self.scale = np.repeat(scale, g)
        self.shape = np.array(shape)
        if self.shape.ndim == 1:
            self.shape = np.tile(shape, (self.g, 1))
        if orientation is None:
            self.orientation = np.tile(np.identity(self.d), (self.g, 1, 1))
        elif orientation.ndim == 2:
            self.orientation = np.tile(orientation, (self.g, 1, 1))
        else:
            self.orientation = orientation

    def get_covariance(self):
        covariance = np.zeros((self.g, self.d, self.d))
        for i in range(self.g):
            covariance[i] = self.scale[i] * \
                            self.orientation[i].dot(np.diag(self.shape[i]))\
                                .dot(self.orientation[i].transpose())

        return covariance

    def select_group(self, index):
        super().select_group(index)
        self.scale = np.repeat(self.scale[index], self.g)
        self.shape = np.tile(self.shape[index], (self.g, 1))
        self.orientation = np.tile(self.orientation[index], (self.g, 1, 1))


class VarianceCholesky(Variance):
    """
    Representation for Covariance structures of a MixtureModel based on
    Cholesky decomposition of covariance matrix.

    Used for EEE and VVV model configurations.
    """
    def __init__(self, d, g, cholsigma):
        """
        Constructor

        :param d: Dimension of the data in MixtureModel
        :param g: Number of cluster components in MixtureModel
        :param cholsigma: NumPy array of 2 or 3 dimension containing the Cholesky decomposition of
                          the covariance matrices of the cluster components. If cholsigma is 3
                          dimensional, the ith element indicates the Cholesky decomposition matrix of
                          cluster component i. If cholsigma is 2 dimensional, the covariance matrices
                          for all cluster components are taken to be equal.
        """
        super().__init__(d, g)
        if cholsigma.ndim == 2:
            self.cholsigma = np.tile(cholsigma, (self.g, 1, 1))
        elif cholsigma.ndim == 3:
            self.cholsigma = cholsigma
        else:
            raise DimensionError("cholsigma should be 2 or 3 dimensional")

    def get_covariance(self):
        result = np.zeros((self.g, self.d, self.d))
        for g in range(self.g):
            result[g] = self.cholsigma[g].transpose().dot(self.cholsigma[g])
        return result

    def select_group(self, index):
        super().select_group(index)
        self.cholsigma = np.tile(self.cholsigma[index], (self.g, 1, 1))
