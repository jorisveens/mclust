import numpy as np
import warnings
from mclust.Exceptions import ModelError, AbstractMethodError
from mclust.fortran.mclust import hcvvv, hceii
from mclust.Utility import traceW, partconv, scale


class HC:
    def __init__(self, data):
        self.data = data.copy(order='F')
        self.pairs = None
        self.partition = None
        self.n = None
        self.p = None
        self.l = None
        self.m = None
        pass

    def fit(self, minclus=1):
        # For now only Mclust default: SVD
        # all values should be finite
        data_scaled = scale(self.data, center=True, rescale=True)
        p = min(data_scaled.shape)
        _, d, v = np.linalg.svd(data_scaled, full_matrices=False)
        z = data_scaled \
            .dot(v.transpose()) \
            .dot(np.diag(np.concatenate((1/np.sqrt(d),
                                         np.zeros(p-len(d)))))
                 )
        z = np.asfortranarray(z)
        self._handle_input(minclus)
        self.hc_fortran(z)

    def _handle_input(self, minclus=1):
        if minclus < 1:
            raise ModelError("minclus must be positive")
        if np.any(np.equal(self.data, None)):
            raise ModelError("Missing values not allowed in data")

        self.n, self.p = self.data.shape
        if self.n <= self.p:
            warnings.warn("Number of observations <= data dimension")

        if self.partition is None:
            self.partition = np.arange(1, self.n+1)
        elif len(self.partition) != self.n:
            raise ModelError("Partition must assign a class to each observation")
        self.partition = partconv(self.partition, consec=True)
        self.l = np.array(len(np.unique(self.partition)), int, order='F')
        self.m = np.array(self.l - minclus, int, order='F')
        if self.m <= 0:
            raise ModelError("initial number of clusters is not greater than minclus")

    def hc_fortran(self, z):
        raise AbstractMethodError()

    def get_class_matrix(self, g):
        n = len(self.partition)
        k = len(np.unique(self.partition))
        g = np.sort(g)[::-1]
        select = k - g
        if len(select) == 1 and select[0] == 0:
            return self.partition.reshape(n, 1)
        bad = np.logical_or(select < 0, select >= k)
        if all(bad):
            raise ModelError("No classification with the specified number of clusters")
        if any(bad):
            warnings.warn("Some selected classifications are inconsistent with mclust objects")

        l = len(select)
        cl = np.full((n, l), float('nan'), float, order='F')
        m = 0
        if select[0] == 0:
            cl[:, 0] = self.partition
            m = 1
        for li in range(max(select)):
            ij = self.pairs[:, li]
            i = min(ij)
            j = max(ij)
            self.partition[self.partition == j] = i
            if select[m] == li + 1:
                cl[:, m] = self.partition
                m = m + 1

        return np.apply_along_axis(partconv, 0, cl[:, range(l-1, -1, -1)])


class HCVVV(HC):
    def hc_fortran(self, z, alpha=1, beta=1):
        ll = int((self.l * (self.l-1))/2)
        ld = np.array(max(self.n, ll + 1, 3 * self.m))
        alpha = np.array(max(alpha * traceW(z / np.sqrt(self.n * self.p)), np.finfo(float).eps), float, order='F')
        beta = np.array(beta, float, order='F')
        data_stacked = np.column_stack((z, np.zeros(self.n)))
        v = np.zeros(self.p, float, order='F')
        u = np.zeros(self.p * self.p, float).reshape(self.p, self.p, order='F')
        s = np.zeros(self.p * self.p, float).reshape(self.p, self.p, order='F')
        r = np.zeros(self.p * self.p, float).reshape(self.p, self.p, order='F')
        d = np.zeros(ld, order='F')
        hcvvv(data_stacked,
              self.partition,
              self.l,
              self.m,
              alpha,
              beta,
              v,
              u,
              s,
              r,
              ld,
              d
              )
        self.pairs = data_stacked[0:self.m, 0:2].transpose()


class HCEII(HC):
    def hc_fortran(self, z):
        ld = np.array(max((self.l * (self.l - 1))/2, 3 * self.m), int, order='F')
        v = np.zeros(self.p, float, order='F')
        d = np.zeros(ld, float, order='F')
        hceii(z,
              np.array(self.p, int, order='F'),
              self.partition,
              self.l,
              self.m,
              v,
              ld,
              d
              )
        self.pairs = z[0:self.m, 0:2].transpose()
