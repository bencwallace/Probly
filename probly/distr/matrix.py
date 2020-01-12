import numpy as np

from probly.lib.utils import array
from probly.distr import Normal, Distribution


class Wigner(Distribution):
    """
    A Wigner random matrix.

    A random symmetric matrix whose upper-diagonal entries are independent,
    identically distributed random variables.

    Parameters
    ----------
    dim : int
        The matrix dimension.
    rv : RandomVariable, optional
        A random variable whose distribution the entries will share. Default is
        a standard normal random variable.
    """

    def __init__(self, dim, rv=None):
        self.dim = dim
        if rv is None:
            self.rv = Normal()
        else:
            self.rv = rv
        arr = [[self.rv.copy() for _ in range(dim)] for _ in range(dim)]
        self.arr = array([[arr[i][j] if i <= j else arr[j][i] for i in range(dim)] for j in range(dim)])

        super().__init__()

    def _sampler(self, seed):
        return self.arr(seed)

    def __str__(self):
        return 'Wigner({}, {})'.format(self.dim, self.rv)


class Wishart(Distribution):
    """
    A Wishart random matrix.

    An `n` by `n` random symmetric matrix obtained as the (matrix) product of
    an `m` by `n` random matrix with independent, identically distributed
    entries, and its transpose.

    Parameters
    ----------
    m : int
        The first dimension parameter.
    n : int
        The second dimension parameter.
    rv : RandomVariable, optional
        A random variable.


    Attributes
    ----------
    lambda_ : float
        The ratio `m / n`.
    """

    def __init__(self, m, n, rv=None):
        self.m = m
        self.n = n
        self.lambda_ = m / n
        if rv is None:
            self.rv = Normal()
        else:
            self.rv = rv
        rect = np.array([[self.rv.copy() for _ in range(n)] for _ in range(m)])
        self.arr = array(np.dot(rect.T, rect))
        super().__init__()

    def _sampler(self, seed):
        return self.arr(seed)

    def __str__(self):
        return 'Wishart({}, {}, {})'.format(self.m, self.n, self.rv)
