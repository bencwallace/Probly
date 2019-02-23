"""probly.py: A python module for working with random variables."""

import copy
import networkx as nx
import numpy as np
import operator as op
import random

# Local
from .helpers import get_seed, _max_seed

# For helpers
from functools import wraps


class rvAbstract(object):
    """
    An abstract random variable placeholder.

    Implements dependency relations and composition map but cannot be sampled
    on its own.
    """

    # Initialize dependency graph
    graph = nx.MultiDiGraph()

    def __init__(self, f=None, *args):
        self.graph.add_node(self, method=f)
        edges = [(self._cast(var), self, {'index': i})
                 for i, var in enumerate(args)]
        self.graph.add_edges_from(edges)

    def __call__(self, seed=None):
        parents = list(self.parents())
        assert len(parents) > 0, 'Independent abstract random variable'\
                                 ' cannot be sampled.'

        seed = get_seed(seed)

        # Create {index: parent} dictionary `arguments`
        data = [self.graph.get_edge_data(p, self) for p in parents]
        arguments = {}
        for i in range(len(parents)):
            indices = [d.values() for d in data[i].values()]
            for j in range(len(indices)):
                arguments[data[i][j]['index']] = parents[i]

        # Sample elements of `parents` in order specified by `arguments`
        # and apply `method` to result
        samples = [arguments[i](seed) for i in range(len(arguments))]
        method = self.graph.nodes[self]['method']
        return method(*samples)

    def parents(self):
        """Returns list of random variables from which `self` is defined"""

        if self in rvGen.graph:
            return list(self.graph.predecessors(self))
        else:
            return []

    @classmethod
    def _cast(cls, obj):
        if isinstance(obj, cls):
            return obj
        else:
            return cls()

    @classmethod
    def Lift(cls, f):
        """
        Lifts a function to the composition map between random variables.

        Args:
            cls (type): Specifies desired output type of lifted map.
        """

        @wraps(f)
        def F(*args):
            """
            The lifted function

            Args:
                `rvar`s and constants
            """

            F_of_args = cls.__new__(cls)

            cls.graph.add_node(F_of_args, method=f)
            edges = [(cls._cast(var), F_of_args, {'index': i})
                     for i, var in enumerate(args)]
            cls.graph.add_edges_from(edges)

            return F_of_args

        return F


# Doesn't need to inherit from rvAbstract?
class rvGen(rvAbstract):
    """A general (concrete) random variable with a sampling method."""

    # Magic
    def __init__(self, sampler=None, origin='random'):
        if sampler is None:
            return

        assert callable(sampler), '{} is not callable'.format(sampler)

        if origin == 'numpy':
            def seeded_sampler(seed=None):
                np.random.seed(seed)
                return sampler()
            self.sampler = seeded_sampler
        elif origin == 'scipy':
            self.sampler = lambda seed=None: sampler.rvs(random_state=seed)
        elif origin == 'random':
            def seeded_sampler(seed=None):
                random.seed(seed)
                return sampler()
            self.sampler = seeded_sampler
        else:
            raise TypeError('Unknown origin `{}`'.format(origin))

    def __call__(self, seed=None):
        if self.sampler is not None:
            seed = get_seed(seed)
            # Seed as follows for independence of `rvGen`s with same `sampler`
            return self.sampler((seed + id(self)) % _max_seed)

    # Instance methods
    def copy(self, obj):
        """Return an independent copy."""

        # Shallow copy is ok as `rvGen` isn't mutable
        return copy.copy(self)

    # Class methods
    @classmethod
    def _cast(cls, obj):
        """Cast constants to `rvGen` objects."""

        if isinstance(obj, cls):
            return obj
        else:
            return cls(lambda seed=None: obj)


class rvNumeric(rvGen):
    """
    A random variable of numeric type. Not for direct use.

    Compatible with numerical operations.
    """

    # Is there a more natural way to inherit the Lift decorator?

    # Left commuting binary operators
    @rvGen.Lift
    def __add__(self, x):
        return op.add(self, x)

    @rvGen.Lift
    def __sub__(self, x):
        return op.sub(self, x)

    @rvGen.Lift
    def __mul__(self, x):
        return op.mul(self, x)

    # Right commuting binary operators
    @rvGen.Lift
    def __radd__(self, x):
        return x.__add__(self)

    @rvGen.Lift
    def __rsub__(self, x):
        return x.__sub__(self)

    @rvGen.Lift
    def __rmul__(self, x):
        return x.__mul__(self)

    # Left non-commuting binary operators
    @rvGen.Lift
    def __matmul__(self, x):
        return op.matmul(self, x)

    @rvGen.Lift
    def __truediv__(self, x):
        return op.truediv(self, x)

    @rvGen.Lift
    def __floordiv__(self, x):
        return op.floordiv(self, x)

    @rvGen.Lift
    def __mod__(self, x):
        return op.mod(self, x)

    @rvGen.Lift
    def __divmod__(self, x):
        return op.divmod(self, x)

    @rvGen.Lift
    def __pow__(self, x):
        return op.pow(self, x)

    # Right non-commuting binary operators
    @rvGen.Lift
    def __rmatmul__(self, x):
        x = rvGen._cast(x)
        return x.__matmul__(self)

    @rvGen.Lift
    def __rtruediv__(self, x):
        x = rvGen._cast(x)
        return x.__truediv__(self)

    @rvGen.Lift
    def __rfloordiv__(self, x):
        x = rvGen._cast(x)
        return x.__floordiv__(self)

    @rvGen.Lift
    def __rmod__(self, x):
        x = rvGen._cast(x)
        return x.__mod__(self)

    @rvGen.Lift
    def __rdivmod__(self, x):
        x = rvGen._cast(x)
        return x.__divmod__(self)

    @rvGen.Lift
    def __rpow__(self, x):
        x = rvGen._cast(x)
        return x.__pow__(self)

    # Unary operators
    @rvGen.Lift
    def __neg__(self):
        return op.neg(self)

    @rvGen.Lift
    def __pos__(self):
        return op.pos(self)

    @rvGen.Lift
    def __abs__(self):
        return op.abs(self)

    @rvGen.Lift
    def __complex__(self):
        return op.complex(self)

    @rvGen.Lift
    def __int__(self):
        return op.int(self)

    @rvGen.Lift
    def __float__(self):
        return op.float(self)

    @rvGen.Lift
    def __round__(self):
        return op.round(self)

    @rvGen.Lift
    def __trunc__(self):
        return op.trunc(self)

    @rvGen.Lift
    def __floor__(self):
        return op.floor(self)

    @rvGen.Lift
    def __ceil__(self):
        return op.ceil(self)


class rvar(rvNumeric):
    """A random scalar."""

    pass


class rarray(rvNumeric):
    """
    A random array.

    Supports subscripting and matrix operations.
    """

    # circular:
    def __new__(cls, arr=None):
        if hasattr(arr, '__getitem__'):
            @cls.Lift
            def make_array(*args):
                return np.array(args)

            arr = np.array([cls._cast(var) for var in arr])
            return make_array(*arr)
        else:
            return super().__new__(cls)

    def __init__(self, arr):
        pass

    # def __getitem__(self, key):
    #     assert hasattr(self(0), '__getitem__'),\
    #         'Scalar {} object not subscriptable'.format(self.__class__)
    #     return rvGen._getitem(self, key)

    # To do: add matrix operations

    # @classmethod
    # def _cast(cls, obj):
    #     """Cast a constant array to a random array."""

    #     if isinstance(obj, cls):
    #         return obj
    #     else:
    #         return rvGen.array(obj)

    # @classmethod
    # def _getitem(cls, obj, key):
    #     def get(arr):
    #         return arr[key]
    #     return cls._compose(get, obj)
