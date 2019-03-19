"""
Additional random variable constructors.
"""

import numpy as np
from collections.abc import Sequence
from functools import partial, wraps
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
    def F(*args, **kwargs):
        if any((isinstance(arg, RandomVar) for arg in args)):
            fkwargs = partial(f, **kwargs)
            return RandomVar(fkwargs, *args)
        else:
            return f(*args, **kwargs)

    return F


def array(arr):
    """
    Casts an array of random variables to a random variable.

    Parameters
    ----------
    arr : array_like
        An array_like collection of `RandomVar` objects or constants.

    Returns
    -------
    RandomVar
        A random variable whose samples are arrays of samples of the objects
        in `arr`.
    """

    def op(*inputs):
        return np.array(inputs).reshape(np.shape(arr))

    parents = np.vectorize(RandomVar)(arr).flatten()
    RV = RandomVar(op, *parents)

    # For RandomVar.__array__
    RV.shape = np.shape(arr)

    return RV


def sum(summands, num=None):
    """
    Sums a collection of random variables.

    If `num` is provided, `summands` is taken to be a single random variable
    and the sum of `num` independent copies of `summands` is returned.
    Otherwise, `summands` is taken to be a collection of random variables or
    random array and its sum is returned.

    Parameters
    ----------
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
