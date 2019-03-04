import numpy as np
from collections.abc import Sequence
from .core import RandomVar


def array(arr):
    """
    Turns an array_like collection of random variables into a random array.

    Parameters
    ----------
    arr (array_like)
        An `array_like` object of `RandomVar` objects, constants, and other
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

        return RandomVar(op, *parents)
    elif isinstance(arr, RandomVar):
        return arr
    else:
        # Treat as constant
        return RandomVar(arr)
