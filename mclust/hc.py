import numpy as np
import warnings
from mclust.Exceptions import ModelError
from mclust.fortran.mclust import hcvvv, hceii, mcltrw

# CONTINUE refactor hc methods
# TODO check small differences with hcvvv output


def hc(data, model_name):
    # For now only Mclust default: SVD
    # all values should be finite
    data_scaled = scale(data, center=True, rescale=True)
    p = min(data_scaled.shape)
    _, d, v = np.linalg.svd(data_scaled, full_matrices=False)
    z = data_scaled\
        .dot(v.transpose())\
        .dot(np.diag(np.concatenate((1/np.sqrt(d),
                                     np.zeros(p-len(d)))))
             )
    return np.asfortranarray(z)


def hclass(hcpairs, g):
    initial = hcpairs["initial"]
    n = len(initial)
    k = len(np.unique(initial))
    g = np.sort(g)[::-1]
    select = k - g
    if len(select) == 1 and select[0] == 0:
        return initial.reshape(n, 1)
    bad = np.logical_or(select < 0, select >= k)
    if all(bad):
        raise ModelError("No classification with the specified number of clusters")
    if any(bad):
        warnings.warn("Some selected classifications are inconsistent with mclust objects")

    l = len(select)
    cl = np.full((n, l), float('nan'), float, order='F')
    m = 0
    if select[0] == 0:
        cl[:, 0] = initial
        m = 1
    for li in range(max(select)):
        ij = hcpairs["pairs"][:, li]
        i = min(ij)
        j = max(ij)
        initial[initial == j] = i
        if select[m] == li + 1:
            cl[:, m] = initial
            m = m + 1
    return np.apply_along_axis(partconv, 0, cl[:, range(l-1, -1, -1)])


def hcVVV(data, partition=None, minclus=1, alpha=1, beta=1):
    if minclus < 1:
        raise ModelError("minclus must be positive")
    if np.any(np.equal(data, None)):
        raise ModelError("Missing values not allowed in data")

    n, p = data.shape
    if n <= p:
        warnings.warn("Number of observations <= data dimension")

    if partition is None:
        partition = np.arange(1, n+1)
    elif len(partition) != n:
        raise ModelError("Partition must assign a class to each observation")
    partition = partconv(partition, consec=True)
    l = np.array(len(np.unique(partition)), int, order='F')
    m = np.array(l - minclus, int, order='F')
    if m <= 0:
        raise ModelError("initial number of clusters is not greater than minclus")

    ll = int((l * (l-1))/2)
    ld = np.array(max(n, ll + 1, 3 * m))
    alpha = np.array(max(alpha * traceW(data / np.sqrt(n * p)), np.finfo(float).eps), float, order='F')
    beta = np.array(beta, float, order='F')
    data_stacked = np.column_stack((data, np.zeros(n)))
    v = np.zeros(p, float, order='F')
    u = np.zeros(p * p, float).reshape(p, p, order='F')
    s = np.zeros(p * p, float).reshape(p, p, order='F')
    r = np.zeros(p * p, float).reshape(p, p, order='F')
    d = np.zeros(ld, order='F')
    hcvvv(data_stacked,
          partition,
          l,
          m,
          alpha,
          beta,
          v,
          u,
          s,
          r,
          ld,
          d,
          )
    data_stacked = data_stacked[0:m, 0:2]
    return {"pairs": data_stacked.transpose(),
            "initial": partition
            }


def hcEII(data, partition=None, minclus=1):
    if minclus < 1:
        raise ModelError("minclus must be positive")
    if np.any(np.equal(data, None)):
        raise ModelError("Missing values not allowed in data")

    n, p = data.shape

    if partition is None:
        partition = np.arange(1, n+1)
    elif len(partition) != n:
        raise ModelError("Partition must assign a class to each observation")
    partition = partconv(partition, consec=True)
    l = np.array(len(np.unique(partition)), int, order='F')
    m = np.array(l - minclus, int, order='F')
    if m <= 0:
        raise ModelError("initial number of clusters is not greater than minclus")
    if n <= p:
        warnings.warn("Number of observations <= data dimension")

    ld = np.array(max((l * (l - 1))/2, 3 * m), int, order='F')
    v = np.zeros(p, float, order='F')
    d = np.zeros(ld, float, order='F')
    hceii(data,
          np.array(p, int, order='F'),
          partition,
          l,
          m,
          v,
          ld,
          d
          )

    data = data[0:m, 0:2]
    return {"pairs": data.transpose(),
            "initial": partition
            }


def traceW(x):
    n, p = x.shape
    p = np.array(p, int, order='F')
    u = np.zeros(p, float, order='F')
    ss = np.array(0, float, order='F')

    mcltrw(x, p, u, ss, n=n)
    return ss


def partconv(x, consec=True):
    n = len(x)
    y = np.zeros(n, int, order='F')
    _, idx = np.unique(x, return_index=True)
    u = x[np.sort(idx)]
    if consec:
        for i in range(len(u)):
            y[x == u[i]] = i
    else:
        for i in u:
            l = x == i
            y[l] = np.arange(9)[l][0]
    return y + 1


def scale(data, center=True, rescale=True):
    data_copy = data.copy(order="F")
    if center:
        data_copy -= data_copy.mean(axis=0)[np.newaxis, :]
    if rescale:
        data_copy /= data_copy.std(axis=0, ddof=1)[np.newaxis, :]
    return data_copy

