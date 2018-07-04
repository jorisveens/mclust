import numpy as np
from math import sqrt, floor, log10
from mclust.Exceptions import ModelError


def qclass(x, k):
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
        nr = len(q)-k-1
        q = q[0:nr]

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
        # UNUSED? miss <- match( groups, u, nomatch = 0) == 0

    if noise:
        noiz = [noisi in groups for noisi in noise]

        if any(noiz):
            raise ModelError("noise incompatible with classification")
        groups = [g for g in groups if g != noise] + [g for g in groups if g == noise]
        noise = int(noise)

    # as.numeric applied to a factor is meaningless
    # groups <- as.numeric(factor(cgroups, levels = unique(cgroups)))
    # classification <- as.numeric(factor(as.character(classification), levels = unique(cgroups)))

    # FIXME as far as I can see this does nothing
    # len(groups) - len(noise) = {len(groups) if noise is False, len(groups) - 1 + 1 if noise is True}
    k = len(groups)

    z = np.zeros((n, k), float, order='F')
    for i in range(k):
        for j in range(n):
            z[j, i] = 1 if classification[j] == groups[i] else 0
    return z


def round_sig(x, sig=3):
    return round(x, sig-int(floor(log10(abs(x))))-1)
