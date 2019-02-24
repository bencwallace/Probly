"""probly.py: A python module for working with random variables."""

import copy
import networkx as nx
import numpy as np
import operator as op
from functools import wraps

# from .programs import _programs
from .helpers import get_seed, _max_seed


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    @wraps(f)
    def F(*args):
        """
        The lifted function

        Args:
            `rv`s and constants
        """

        return rv(f, *args)

    return F


class Numeric(object):
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


class rv(Numeric):
    """
    A random variable placeholder.

    Can be acted upon by arithmetical operations and functions compatible with
    `Lift`.
    """

    graph = nx.MultiDiGraph()

    def __init__(self, method=None, *args):
        # assert callable(method), '{} is not callable'.format(method)

        if len(args) == 0:
            args = [root]
        if method is None:
            # Seeding wrapper for `self.sampler`
            def sampler(seed=None):
                np.random.seed(seed)
                return self.sampler(seed)
            method = sampler

        edges = [(rv._cast(var), self, {'index': i})
                 for i, var in enumerate(args)]
        rv.graph.add_node(self, method=method)
        rv.graph.add_edges_from(edges)

    def __call__(self, seed=None):
        seed = get_seed(seed)
        parents = list(self.parents())

        # Create {index: parent} dictionary `arguments`
        # This could probably be made more clear
        data = [rv.graph.get_edge_data(p, self) for p in parents]
        arguments = {}
        for i in range(len(parents)):
            indices = [d.values() for d in data[i].values()]
            for j in range(len(indices)):
                arguments[data[i][j]['index']] = parents[i]

        # Sample elements of `parents` in order specified by `arguments`
        # and apply `method` to result
        samples = [arguments[i]((seed + id(self)) % _max_seed)
                   for i in range(len(arguments))]
        method = rv.graph.nodes[self]['method']
        return method(*samples)

    def __getitem__(self, key):
        assert hasattr(self(0), '__getitem__'),\
            'Scalar {} object not subscriptable'.format(self.__class__)
        return rv._getitem(self, key)

    def parents(self):
        """Returns list of random variables from which `self` is defined"""
        if self in rv.graph:
            return list(rv.graph.predecessors(self))
        else:
            return []

    # Constructors
    @classmethod
    def _getitem(cls, obj, key):
        def get(arr):
            return arr[key]
        return Lift(get)(obj)

    @classmethod
    def _cast(cls, obj):
        """Cast constants to `rv` objects."""

        if isinstance(obj, cls):
            return obj
        elif hasattr(obj, '__getitem__'):
            return rv.array(obj)
        else:
            return cls(lambda seed=None: obj)

    @classmethod
    def copy(cls, obj):
        """Return a random variable with the same distribution as `self`"""

        # Shallow copy is ok as `rv` isn't mutable
        return copy.copy(obj)

    @classmethod
    def array(cls, arr):
        arr = np.array([rv._cast(var) for var in arr])

        def make_array(*args):
            return np.array(args)
        return Lift(make_array)(*arr)


class Root(rv):
    """The root of the dependency tree is the random number generator."""

    def __init__(self):
        rv.graph.add_node(self)

    def __call__(self, seed=None):
        return get_seed(seed)


root = Root()
