"""
Random variables
================

This modules defines `RandomVar` objects as nodes in the dependency graph
(`Node` objects) with extra structure allowing them to be transformed and
operated on in whatever way is compatible with their realizations.
"""

import math
import numpy as np
from functools import wraps
import operator as op

from .core import Node


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
        if any([isinstance(arg, RandomVar) for arg in args]):
            return RandomVar(f, *args)
        else:
            # Could be a problem if F(*args)() called
            return f(*args)

    return F


def array(arr):
    """
    Turns a collection of random variables and constants into a random array.

    Parameters
    ----------
    arr (array_like)
        An `array_like` object of `RandomVar` objects, constants, and other
        `array_like` objects.

    Returns
    -------
    RandomVar
        A random variable whose samples are arrays of samples of the objects in
        `arr`.
    """

    arr = [RandomVar._cast(var) for var in arr]

    @Lift
    def make_array(*args):
        return np.array(args)

    return make_array(*arr)


class RandomVar(Node):
    """
    A random variable.

    Can be acted upon by in the following ways (when its samples can):
    - By functions decorated with `Lift`;
    - By arithmetical operations (when its values can);
    - By subscripting; and
    - As an iterator.
    """

    # Sequence and array magic
    def __array__(self):
        parents = self.parents()
        return np.array(parents, dtype=object)

    def __getitem__(self, key):
        key = self._cast(key)

        @Lift
        def get(arr, *args):
            return arr[args]
        return get(self, *key)

    def __iter__(self):
        self.index = 0
        self.ordered_parents = self.parents()
        return self

    def __next__(self):
        if self.index < len(self.ordered_parents):
            p = self.ordered_parents[self.index]
            self.index += 1
            return p
        else:
            raise StopIteration

    @classmethod
    def _cast(cls, obj):
        """Cast constants to `Const` objects."""

        if isinstance(obj, cls):
            return obj
        elif hasattr(obj, '__getitem__'):
            return array(obj)
        else:
            return Const(obj)

    # Matrix operators
    # To do: check how these are really used
    @Lift
    def __matmul__(self, x):
        return op.matmul(self, x)

    def __rmatmul__(self, x):
        X = self._cast(x)
        return X.__matmul__(self)

    # Left commuting binary operators
    @Lift
    def __add__(self, x):
        return self + x

    @Lift
    def __sub__(self, x):
        return self - x

    @Lift
    def __mul__(self, x):
        return self * x

    # Right commuting binary operators
    def __radd__(self, x):
        X = self._cast(x)
        return X + self

    def __rsub__(self, x):
        X = self._cast(x)
        return X - self

    def __rmul__(self, x):
        X = self._cast(x)
        return X * self

    # Left non-commuting binary operators
    @Lift
    def __truediv__(self, x):
        return self / x

    @Lift
    def __floordiv__(self, x):
        return self // x

    @Lift
    def __mod__(self, x):
        return self % x

    @Lift
    def __divmod__(self, x):
        return divmod(self, x)

    @Lift
    def __pow__(self, x):
        return self ** x

    # Right non-commuting binary operators
    def __rtruediv__(self, x):
        X = self._cast(x)
        return X / self

    def __rfloordiv__(self, x):
        X = self._cast(x)
        return X // self

    def __rmod__(self, x):
        X = self._cast(x)
        return X % self

    def __rdivmod__(self, x):
        X = self._cast(x)
        return divmod(X, self)

    def __rpow__(self, x):
        X = self._cast(x)
        return X ** self

    # Unary operators
    @Lift
    def __neg__(self):
        return -self

    @Lift
    def __pos__(self):
        return +self

    @Lift
    def __abs__(self):
        return abs(self)

    @Lift
    def __complex__(self):
        return complex(self)

    @Lift
    def __int__(self):
        return int(self)

    @Lift
    def __float__(self):
        return float(self)

    @Lift
    def __round__(self):
        return round(self)

    @Lift
    def __trunc__(self):
        return math.trunc(self)

    @Lift
    def __floor__(self):
        return math.floor(self)

    @Lift
    def __ceil__(self):
        return math.ceil(self)


class Const(RandomVar):
    def __init__(self, value):
        self.value = value

    def __call__(self, seed=None):
        return self.value

    def __str__(self):
        return 'const {}'.format(self.value)
