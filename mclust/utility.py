from math import sqrt, floor, log10

import numpy as np
from mclust.fortran.mclust import mcltrw

from mclust.exceptions import ModelError


def qclass(x, k):
    """
    Initialisation method for clustering for one dimensional data based on frequency .
    :param x: Data to cluster.
    :param k: Amount of groups in clustering.
    :return: Basic clustering based on data frequency.
    """
    x_flat = x.flatten()
    # eps <- sqrt(.Machine$double.eps)
    # numerical accuracy problem if scale of x is large, so make tolerance
    # scale dependent
    eps = x_flat.std(ddof=1)*sqrt(np.finfo(float).eps)
    q = []
    n = k
    while len(q) < (k+1):
        n = n + 1
        q = list(set(map(lambda p: np.percentile(x_flat, p * 100), [p/(n-1) for p in range(n)])))
        q.sort()

    if len(q) > (k+1):
        # only take largest quantiles
        dq = np.diff(q)

        nr = len(q)-k-1
        select = np.argsort(dq)[0:nr]
        q = np.delete(q, select)

    q[0] = min(x_flat) - eps
    q[-1] = max(x_flat) + eps
    cl = np.repeat(0, len(x_flat))
    for i in range(k):
        # set cl[i] to be i if it is between the ith and ith + 1 quantile
        for index, xi in enumerate(x_flat):
            if q[i] <= xi < q[i + 1]:
                cl[index] = i

    return cl


def mclust_map(z):
    """
    Converts a matrix in which each row sums to 1 to an integer vector
    specifying for each row the column index of the maximum.

    :param z: A matrix (for example a matrix of conditional probabilities
              in which each row sums to 1 as produced by the E-step of the
              EM algorithm).
    :return: A integer vector with one entry for each row of z, in which the
             i-th value is the column index at which the i-th row of z
             attains a maximum.
    """
    return np.argmax(z, axis=1)


def mclust_unmap(classification, groups=None, noise=None):
    """
    converts a classification to conditional probabilities
    classes are arranged in sorted order unless groups is specified
    if a noise indicator is specified, that column is placed last.
    """
    n = len(classification)
    u = sorted(set(classification))
    if groups is None:
        groups = u
    else:
        if any([not (ui in groups) for ui in u]):
            raise ModelError("groups incompatible with classification")

    if noise:
        noiz = [noisi in groups for noisi in noise]

        if any(noiz):
            raise ModelError("noise incompatible with classification")
        groups = [g for g in groups if g != noise] + [g for g in groups if g == noise]

    k = len(groups)

    z = np.zeros((n, k), float, order='F')
    for i in range(k):
        for j in range(n):
            z[j, i] = 1 if classification[j] == groups[i] else 0
    return z


def round_sig(x, sig=3):
    """
    Round x to sig significant digits
    :param x: Number to round
    :param sig: Number of significant digits
    :return: rounded number.
    """
    return round(x, sig-int(floor(log10(abs(x))))-1)


def traceW(x):
    n, p = x.shape
    p = np.array(p, int, order='F')
    u = np.zeros(p, float, order='F')
    return mcltrw(x, p, u, n=n)


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
    """
    centers and/or scales the columns of a numeric matrix.

    :param data: Data matrix to center and/or scale
    :param center: Boolean indicating if data should be centered around 0.
    :param rescale: Boolean indication if data should be rescaled to have
                    standard deviation of 1.
    :return: Centered and/or scaled data.
    """
    data_copy = data.copy(order="F")
    if center:
        data_copy -= data_copy.mean(axis=0)[np.newaxis, :]
    if rescale:
        data_copy /= data_copy.std(axis=0, ddof=1)[np.newaxis, :]
    return data_copy
