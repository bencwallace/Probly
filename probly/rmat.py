"""
Random matrix constructors.
"""

import numpy as np

from .core import RandomVar
from .distr import Normal
from .helpers import array


class Wigner(RandomVar):
    """
    A Wigner random matrix.

    A random symmetric matrix whose upper-diagonal entries are independent,
    identically distributed random variables.

    Parameters
    ----------
    dim : int
        The matrix dimension.
    rv : RandomVar, optional
        A random variable whose distribution the entries will share. Default is
        a standard normal random variable.
    """

    def __new__(cls, dim, rv=None):
        if rv is None:
            rv = Normal()

        # Upper-diagonal part
        arr = [[rv.copy() if i <= j else 0 for j in range(dim)]
               for i in range(dim)]

        # Lower-diagonal part
        arr = [[arr[i][j] if i <= j else arr[j][i] for j in range(dim)]
               for i in range(dim)]

        rarr = array(arr)
        rarr.__class__ = cls
        return rarr

    def __init__(self, dim, rv=None):
        self.dim = dim


class Wishart(RandomVar):
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
    rv : RandomVar, optional
        A random variable.


    Attributes
    ----------
    lambda_ : float
        The ratio `m / n`.
    """

    def __new__(cls, m, n, rv=None):
        if rv is None:
            rv = Normal()

        rect = np.array([[rv.copy() for j in range(n)] for i in range(m)])
        square = np.dot(rect.T, rect)

        rarr = array(square)
        rarr.__class__ = cls
        return rarr

    def __init__(self, m, n, rv=None):
        self.m = m
        self.n = n
        self.lambda_ = self.m / self.n
