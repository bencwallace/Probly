"""probly.py: A python module for working with random variables."""

import copy
import numpy as np
import operator as op
import math
from functools import wraps
import itertools

# from .programs import _programs
from .helpers import get_seed, _max_seed

# Initialize dependency graph
import probly.graphtools as gt


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    @wraps(f)
    def F(*args):
        if any([isinstance(arg, rv) for arg in args]):
            return rv(f, *args)
        else:
            # Could be a problem if F(*args)() called
            return f(*args)

    return F


def array(arr):
    """Turn an array of `rv` objects and constants into a random variable."""

    # arr = np.array([rv._cast(var) for var in arr])
    arr = [rv._cast(var) for var in arr]

    @Lift
    def make_array(*args):
        return np.array(args)

    return make_array(*arr)


class rv(object):
    """
    A random variable.

    Can be acted upon by functions decorated with `Lift`. Can also be acted
    upon by numerical operations when its values can.
    """

    # Track random variables for independence
    _last_id = itertools.count()

    # Core magic methods
    def __new__(cls, function=None, *args):
        """
        Creates a random variable object.

        The new random variable is function(*args) if these are non-emtpy.

        Args:
            function (function): The function being applied.
            args (list): A list of random variables or constants.
        """

        # Build graph regardless of how rv was created
        obj = super().__new__(cls)
        edges = [(rv._cast(var), obj, {'index': i})
                 for i, var in enumerate(args)]
        gt._graph.add_node(obj)
        gt._graph.add_edges_from(edges)

        return obj

    def __init__(self, function=None, *args):
        self._id = next(self._last_id)
        self.function = function

    def __call__(self, seed=None):
        seed = get_seed(seed)
        parents = self.parents()

        if len(parents) == 0:
            # if seed is None:
            #     seed = get_seed(seed)
            return self.sampler((seed + self._id) % _max_seed)
        samples = [parents[i](seed)
                   for i in range(len(parents))]
        return self.function(*samples)

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

    # Representation magic
    def __str__(self):
        return 'RV {}'.format(self._id)

    # Helper methods
    def parents(self):
        """Returns list of random variables from which `self` is defined"""
        if self not in gt._graph:
            return []
        else:
            # Convert to list for re-use
            unordered = list(gt._graph.predecessors(self))
            if len(unordered) == 0:
                return []

            data = [gt._graph.get_edge_data(p, self) for p in unordered]

            # Create {index: parent} dictionary
            dictionary = {}
            for i in range(len(unordered)):
                indices = [d.values() for d in data[i].values()]
                for j in range(len(indices)):
                    dictionary[data[i][j]['index']] = unordered[i]

            ordered = [dictionary[i] for i in range(len(dictionary))]
            return ordered

    def copy(self):
        """Return a random variable with the same distribution as `self`"""

        # Alternative is to define __copy__

        # Shallow copy is ok as `rv` isn't mutable
        Copy = copy.copy(self)
        # Need to update id manually when copying
        Copy._id = next(Copy._last_id)

        return Copy

    @staticmethod
    def _cast(obj):
        """Cast constants to `Const` objects."""

        if isinstance(obj, rv):
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


class Const(rv):
    def __init__(self, value):
        self.value = value

    def __call__(self, seed=None):
        return self.value

    def __str__(self):
        return 'const {}'.format(self.value)
