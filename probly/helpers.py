"""
Additional methods for random variable formation.
"""


import numpy as np
from collections.abc import Sequence
from functools import wraps
from .core import RandomVar


def Lift(f):
    """
    "Lifts" and returns a function to the composition map between random
    variables.

    Can be used as a decorator.

    Note
    ----
    Functions that manipulate their arguments using already lifted functions
    (such as arithmetical operations) do not need to be lifted themselves.
    For instance, the following example, which does not make use of `Lift`,
    works without issue:

    >>> import probly as pr
    >>> X = pr.Unif(0, 1)
    >>> M = pr.array([[X, X + 10], [X + 100, X + 1000]])
    >>> def f(x):
    ...     return x[0, 0] - x[1, 1]
    >>> Y = f(M)
    >>> print(Y())
    -1000
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
        An `array_like` collection of `RandomVar` objects, constants, and other
        `array_like` objects.

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

        # Trying to get np.linalg.det to work
        RV._isarray = True

        return RV
    elif isinstance(arr, RandomVar):
        return arr
    else:
        # Treat as constant
        return RandomVar(arr)
