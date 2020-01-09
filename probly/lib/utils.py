"""
Utilities for working with random variables.
"""

from functools import partial, wraps

import matplotlib.pyplot as plt
import numpy as np

from ..core.random_variables import RandomVariable
from ..core.random_arrays import RandomArray


def const(c):
    """
    Constructs a constant random variable.

    :param c: constant value
    :return: Constant random variable with value `c`
    """

    return RandomVariable(lambda _: c)


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


# todo: summing dimension
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
        # assume summands is a random variable
        summands = RandomArray([summands.copy() for _ in range(num)])
    else:
        # Assume summands is a collection of random variables or random array
        summands = RandomArray(summands)

    return np.sum(summands)
