"""
Additional methods for random variable formation.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from functools import wraps
from .core import RandomVar


def Lift(f):
    """
    Lifts a function to the composition map between random variables.

    Can be used as a decorator.


    Example
    -------
    Construct a random variable given by the determinant of a Wigner matrix.

    .. testsetup::

       import probly as pr
       from probly.core import RandomVar
       RandomVar._reset()


    >>> import numpy as np
    >>> Det = pr.Lift(np.linalg.det)
    >>> M = pr.Wigner(pr.Normal(), 2)
    >>> D = Det(M)
    >>> D(0)
    1.2638214666689431
    """

    @wraps(f)
    def F(*args):
        if any((isinstance(arg, RandomVar) for arg in args)):
            return RandomVar(f, *args)
        else:
            return f(*args)

    return F


def array(arr):
    """
    Casts an array of random variables to a random variable.

    Parameters
    ----------
    arr : array_like
        A collection of `RandomVar` objects, constants, and other
        array_like objects.

    Returns
    -------
    RandomVar
        A random variable whose samples are arrays of samples of the objects
        in `arr`.
    """

    if isinstance(arr, (Sequence, np.ndarray)):
        def op(*inputs):
            return np.array(inputs)

        # Recursively turn elements into RandomVar objects
        parents = (array(p) for p in arr)

        RV = RandomVar(op, *parents)

        # For RandomVar.__array__
        RV._isarray = True

        return RV
    elif isinstance(arr, RandomVar):
        return arr
    else:
        # Treat as constant
        return RandomVar(arr)


def hist(rv, num_samples, bins=20, density=True):
    """
    Plots a histogram from samples of a random variable.

    Parameters
    ----------
    rv : RandomVar
        A random variable.
    num_samples : int
        The number of samples to draw from `rv`.
    bins : int
        The number of bins in the histogram.
    density : bool
        If True, the histogram is normalized to form a probability density.
    """

    samples = [rv() for _ in range(num_samples)]
    plt.hist(samples, bins=bins, density=density)
    plt.show()


def sum(summands, num=None):
    """
    Sums a collection of random variables.

    If `num` is provided, `summands` is taken to be a single random variable
    and the sum of `num` independent copies of `summands` is returned.
    Otherwise, `summands` is taken to be a collection of random variables or
    random array and its sum is returned.

    summands : array_like of RandomVar or RandomVar
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
