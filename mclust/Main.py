from mclust.fortran import mclustaddson
from mclust.fortran import mclust
import numpy as np


def dummy() -> int:
    """
    Dummy function that always returns 1.
    :return: 1
    """
    return 1


if __name__ == '__main__':
    x = np.arange(9.).reshape(3, 3, order='F')
    y = np.arange(9.).reshape(3, 3, order='F')
    z = np.arange(9.).reshape(3, 3, order='F')
    print(x)
    mclustaddson.transpose(x)
    print(x)

    print("-------------")
    mclustaddson.crossprodf(x, y, z)
    print(x)
    print(y)
    print(z)

    print(mclust.__doc__)
    print(mclust.dgamma.__doc__)
    print(mclust.dgamma(0.5))
