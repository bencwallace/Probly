"""probly.py: A python module for working with random variables."""

import copy
import networkx as nx
import numpy as np
import operator as op
from functools import wraps
import itertools

# from .programs import _programs
from .helpers import get_seed, _max_seed


# Initialize global dependency graph
graph = nx.MultiDiGraph()


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
        graph.add_node(obj)
        graph.add_edges_from(edges)

        return obj

    def __init__(self, function=None, *args):
        self._id = next(self._last_id)
        self.function = function

    def __call__(self, seed=None):
        parents = list(self.parents())
        if len(parents) == 0:
            if seed is None:
                seed = get_seed(seed)
            return self.sampler_fixed((seed + self._id) % _max_seed)

        data = [graph.get_edge_data(p, self) for p in parents]

        # Create {index: parent} dictionary
        inputs = {}
        for i in range(len(parents)):
            indices = [d.values() for d in data[i].values()]
            for j in range(len(indices)):
                inputs[data[i][j]['index']] = parents[i]

        # Sample elements of parents in order specified by inputs
        # and apply function to result
        samples = [inputs[i](seed)
                   for i in range(len(inputs))]
        return self.function(*samples)

    def __getitem__(self, key):
        # assert hasattr(self(0), '__getitem__'),\
        #     'Scalar {} object not subscriptable'.format(self.__class__)

        @Lift
        def get(arr):
            return arr[key]
        return get(self)

    def parents(self):
        """Returns list of random variables from which `self` is defined"""
        if self in graph:
            return list(graph.predecessors(self))
        else:
            return []

    def copy(self):
        """Return a random variable with the same distribution as `self`"""

        # Shallow copy is ok as `rv` isn't mutable
        c = copy.copy(self)
        # Need to update id manually when copying
        c._id = next(c._last_id)
        return c

    def sampler_fixed(self):
        # Overload in .distr.Distr
        pass

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
    @Lift
    def __matmul__(self, x):
        return op.matmul(self, x)

    def __rmatmul__(self, x):
        X = self._cast(x)
        return X.__matmul__(self)

    # Left commuting binary operators
    @Lift
    def __add__(self, x):
        return op.add(self, x)

    @Lift
    def __sub__(self, x):
        return op.sub(self, x)

    @Lift
    def __mul__(self, x):
        return op.mul(self, x)

    # Right commuting binary operators
    def __radd__(self, x):
        X = self._cast(x)
        return X.__add__(self)

    def __rsub__(self, x):
        X = self._cast(x)
        return X.__sub__(self)

    def __rmul__(self, x):
        X = self._cast(x)
        return X.__mul__(self)

    # Left non-commuting binary operators
    @Lift
    def __truediv__(self, x):
        return op.truediv(self, x)

    @Lift
    def __floordiv__(self, x):
        return op.floordiv(self, x)

    @Lift
    def __mod__(self, x):
        return op.mod(self, x)

    @Lift
    def __divmod__(self, x):
        return op.divmod(self, x)

    @Lift
    def __pow__(self, x):
        return op.pow(self, x)

    # Right non-commuting binary operators
    def __rtruediv__(self, x):
        X = self._cast(x)
        return X.__truediv__(self)

    def __rfloordiv__(self, x):
        X = self._cast(x)
        return X.__floordiv__(self)

    def __rmod__(self, x):
        X = self._cast(x)
        return X.__mod__(self)

    def __rdivmod__(self, x):
        X = self._cast(x)
        return X.__divmod__(self)

    def __rpow__(self, x):
        X = self._cast(x)
        return X.__pow__(self)

    # Unary operators
    @Lift
    def __neg__(self):
        return op.neg(self)

    @Lift
    def __pos__(self):
        return op.pos(self)

    @Lift
    def __abs__(self):
        return op.abs(self)

    @Lift
    def __complex__(self):
        return op.complex(self)

    @Lift
    def __int__(self):
        return op.int(self)

    @Lift
    def __float__(self):
        return op.float(self)

    @Lift
    def __round__(self):
        return op.round(self)

    @Lift
    def __trunc__(self):
        return op.trunc(self)

    @Lift
    def __floor__(self):
        return op.floor(self)

    @Lift
    def __ceil__(self):
        return op.ceil(self)


class Const(rv):
    def __init__(self, value):
        self.value = value

    def __call__(self, seed=None):
        return self.value
