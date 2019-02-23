"""probly.py: A python module for working with random variables."""

import copy
import networkx as nx
import numpy as np
import operator as op
import random

# Local
from .helpers import get_seed, _max_seed, Lift

# For helpers
from functools import wraps


class rvGen(object):
    """
    An abstract random variable placeholder.

    Implements dependency relations and composition map but cannot be sampled
    on its own.
    """

    # Initialize dependency graph
    graph = nx.MultiDiGraph()

    def __init__(self, sampler=None, origin='random'):
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
            raise TypeError('Sampler {} from {}'
                            ' unknown.'.format(sampler, origin))

    def __call__(self, seed=None):
        seed = get_seed(seed)
        parents = self.parents()

        if len(parents) == 0:
            # Seed as follows for independence of `rvGen`s with same `sampler`
            return self.sampler((seed + id(self)) % _max_seed)

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

    # Instance methods
    def copy(self, obj):
        """Return an independent copy."""

        # Shallow copy is ok as `rvGen` isn't mutable
        return copy.copy(self)

    def parents(self):
        """Returns list of random variables from which `self` is defined"""

        if self in rvGen.graph:
            return list(self.graph.predecessors(self))
        else:
            return []

    # Class methods
    @classmethod
    def _cast(cls, obj):
        if isinstance(obj, cls):
            return obj
        else:
            return cls(lambda seed=None: obj)


class rvNumeric(rvGen):
    """
    A random variable of numeric type. Not for direct use.

    Compatible with numerical operations.
    """

    def __add__(self, x):
        return Lift(type(self), op.add)(self, x)


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
            @Lift
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
