"""
Utilities for working with random variables.
"""

from functools import partial, wraps

import matplotlib.pyplot as plt
import numpy as np

from .core import RandomVariable
from .distributions import Normal


# ------------------------ Random matrix constructors ------------------------ #


class Wigner(RandomVariable):
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
        if rv is None:
            self.rv = Normal()
        else:
            self.rv = rv

    def __str__(self):
        return 'Wigner({}, {})'.format(self.dim, self.rv)


class Wishart(RandomVariable):
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

    def __new__(cls, m, n, rv=None):
        if rv is None:
            rv = Normal()

        rect = np.array([[rv.copy() for _ in range(n)] for _ in range(m)])
        square = np.dot(rect.T, rect)

        rarr = array(square)
        rarr.__class__ = cls
        return rarr

    def __init__(self, m, n, rv=None):
        self.m = m
        self.n = n
        self.lambda_ = self.m / self.n
        if rv is None:
            self.rv = Normal()
        else:
            self.rv = rv

    def __str__(self):
        return 'Wishart({}, {}, {})'.format(self.m, self.n, self.rv)


# ---------------------------- Other constructors ---------------------------- #


def array(arr):
    """
    Casts an array of random variables to a random variable.

    Parameters
    ----------
    arr : array_like
        An array_like collection of `RandomVariable` objects or constants.

    Returns
    -------
    RandomVariable
        A random variable whose samples are arrays of samples of the objects
        in `arr`.
    """

    def op(*inputs):
        return np.array(inputs).reshape(np.shape(arr))

    parents = np.vectorize(RandomVariable)(arr).flatten()
    rv = RandomVariable(op, *parents)

    # For RandomVariable.__array__
    rv.shape = np.shape(arr)

    return rv


def constrv(c):
    """
    Constructs a constant random variable.

    :param c: constant value
    :return: Constant random variable with value `c`
    """

    return RandomVariable(lambda _: c)

def lift(f):
    """
    Lifts a function to the composition map between random variables.

    Can be used as a decorator.

    Example
    -------
    Construct a random variable given by the determinant of a Wigner matrix.

    .. testsetup::

       import probly as pr
       from probly.core import RandomVariable
       RandomVariable._reset()


    >>> import numpy as np
    >>> Det = pr.Lift(np.linalg.det)
    >>> M = pr.Wigner(pr.Normal(), 2)
    >>> D = Det(M)
    >>> D(0)
    1.2638214666689431
    """

    @wraps(f)
    def lifted(*args, **kwargs):
        if any((isinstance(arg, RandomVariable) for arg in args)):
            fkwargs = partial(f, **kwargs)
            return RandomVariable(fkwargs, *args)
        else:
            return f(*args, **kwargs)

    return lifted


def sum(summands, num=None):
    """
    Sums a collection of random variables.

    If `num` is provided, `summands` is treated as a single random variable
    and the sum of `num` independent copies of `summands` is returned.
    Otherwise, `summands` treated as a collection of random variables or
    a random array and its sum is returned.

    Parameters
    ----------
    summands : array_like of RandomVariable or RandomVariable
        Either a collection of random variables to be summed, a random array
        whose entries are to be summed, or a single random variable, of which
        independent copies are to be summed.
    num : int, optional
        The number of independent copies of a random variable to sum.
    """

    if num is not None:
        # Assume summands is a random variable
        summands = array([summands.copy() for _ in range(num)])
    else:
        # Assume summands is a collection of random variables or random array
        summands = array(summands)

    return np.sum(summands)


# -------------------------------- Properties -------------------------------- #


def mean(rv, max_iter=int(1e5), tol=1e-5):
    return rv.mean(max_iter=max_iter, tol=tol)


def moment(rv, p, **kwargs):
    return rv.moment(p, **kwargs)


def cmoment(rv, p, **kwargs):
    return rv.cmoment(p, **kwargs)


def variance(rv, **kwargs):
    return rv.variance(**kwargs)


def cdf(rv, x, **kwargs):
    return rv.cdf(x, **kwargs)


def pdf(rv, x, dx=1e-5, **kwargs):
    return rv.pdf(x, dx, **kwargs)


# ---------------------------------- Other ---------------------------------- #


def hist(rv, num_samples, bins=None, density=True):
    """
    Plots a histogram from samples of a random variable.

    Parameters
    ----------
    rv : RandomVariable
        A random variable.
    num_samples : int
        The number of samples to draw from `rv`.
    bins : int or sequence, optional
        Specifies the bins in the histogram.
    density : bool, optional
        If True, the histogram is normalized to form a probability density.
    """

    samples = [rv() for _ in range(num_samples)]
    plt.hist(samples, bins=bins, density=density)
    plt.show()
