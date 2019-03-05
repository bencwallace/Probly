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

    Example
    -------
    Consider the following custom-built random class:

    >>> import numpy as np
    >>> import string
    >>> import probly as pr
    >>> charset = list(string.ascii_letters)
    >>> class RandomString(pr.Distr):
    ... def __init__(self, rate):
    ...     self.rate = rate
    ...     def _sampler(self, seed=None):
    ...         sample = ''
    ...         Length = pr.Pois(self.rate)(seed=seed)
    ...         for i in range(Length):
    ...             if pr.Ber(0.2)(seed + i) == 1:
    ...                 sample += ' '
    ...             else:
    ...                 np.random.seed(seed + i)
    ...                 char = np.random.choice(charset)
    ...                 sample += char
    ...         return sample
    >>> S = RandomString(20)
    >>> print(S())

    The following example lifts the `str.title` method:

    >>> Title = pr.Lift(str.title)
    >>> T = Title(S)
    >>> print(T())

    In the next example, we decorate a custom function to get a lifted
    function:

    >>> @pr.Lift
    >>> def title_first_char(s):
    ...     return s.title()[0]
    >>> U = title_first_char(S)
    >>> print(T())
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

        RV = RandomVar(op, *parents)

        # Trying to get np.linalg.det to work
        RV._array = True

        return RV
    elif isinstance(arr, RandomVar):
        return arr
    else:
        # Treat as constant
        return RandomVar(arr)
