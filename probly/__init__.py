"""probly.py: A python module for working with random variables."""

import copy
import networkx as nx
import numpy as np
import operator as op

import random

from .programs import _programs
from .helpers import Lift, get_seed


# Numpy max seed
_max_seed = 2 ** 32 - 1


class rvar(object):
    """
    A general random variable.

    Can be sampled from and acted upon by appropriate functions decorated with
    `Lift`.
    """

    graph = nx.MultiDiGraph()

    def __init__(self, sampler=None, f=None, *args):
        self.sampler = sampler

        if self.sampler is not None:
            assert callable(sampler), 'Sampler {} is not'\
                                      'callable'.format(sampler)
        elif f is not None:
            rvar.graph.add_node(self, method=f)
            edges = [(rvar._cast(var), self, {'index': i})
                     for i, var in enumerate(args)]
            rvar.graph.add_edges_from(edges)

    def __call__(self, seed=None):
        seed = get_seed(seed)
        parents = list(self.parents())

        if len(parents) == 0:
            # Seed as follows for independence of `rvar`s with same `sampler`
            return self.sampler((seed + id(self)) % _max_seed)
        else:
            # Create {index: parent} dictionary `arguments`
            data = [rvar.graph.get_edge_data(p, self) for p in parents]
            arguments = {}
            for i in range(len(parents)):
                indices = [d.values() for d in data[i].values()]
                for j in range(len(indices)):
                    arguments[data[i][j]['index']] = parents[i]

            # Sample elements of `parents` in order specified by `arguments`
            # and apply `method` to result
            samples = [arguments[i](seed) for i in range(len(arguments))]
            method = rvar.graph.nodes[self]['method']
            return method(*samples)

    def parents(self):
        """Returns list of random variables from which `self` is defined"""

        if self in rvar.graph:
            return list(rvar.graph.predecessors(self))
        else:
            return []

    # Constructors
    @classmethod
    def _compose(cls, f, *args):
        return cls(None, f, *args)

    @classmethod
    def _cast(cls, obj):
        """Cast constants to `rvar` objects."""

        if isinstance(obj, cls):
            return obj
        else:
            return cls(lambda seed=None: obj)

    @staticmethod
    def array(cls, obj):
        """Construct rvarArray object."""

        return rvarArray(cls, obj)

    @classmethod
    def copy(cls, obj):
        """Return a random variable with the same distribution as `self`"""

        # Shallow copy is ok as `rvar` isn't mutable
        return copy.copy(obj)

    @classmethod
    def define(cls, obj, origin=None):
        """Define a random variable from a pre-defined sampler"""

        if origin is None:
            return cls(obj)
        # Following are obsolete for this class
        elif origin == 'scipy':
            return cls(lambda seed=None: obj.rvs(random_state=seed))
        elif origin == 'numpy':
            def seeded_sampler(seed=None):
                np.random.seed(seed)
                return obj()
            return cls(seeded_sampler)
        elif origin == 'random':
            def seeded_sampler(seed=None):
                random.seed(seed)
                return obj()
            return cls(seeded_sampler)
        else:
            raise TypeError('Unknown origin `{}`'.format(origin))


class rvarNumeric(rvar):
    """
    A random variable of numeric type. Not for direct use.

    Compatible with numerical operations.
    """

    # Define operators for emulating numeric types
    for p in _programs:
        exec(p)


class rvarScalar(rvarNumeric):
    """A random scalar."""

    pass


class rvarArray(rvarNumeric):
    """
    A random array.

    Supports subscripting and matrix operations.
    """

    def __init__(self, arr):
        arr = np.array([rvar._cast(var) for var in arr])

        def make_array(*args):
            return np.array(args)
        return _compose(make_array, *arr)

    def __getitem__(self, key):
        assert hasattr(self(0), '__getitem__'),\
            'Scalar {} object not subscriptable'.format(self.__class__)
        return rvar._getitem(self, key)

    # To do: add matrix operations

    @classmethod
    def _cast(cls, obj):
        """Cast a constant array to a random array."""

        if isinstance(obj, cls):
            return obj
        else:
            return rvar.array(obj)

    @classmethod
    def _getitem(cls, obj, key):
        def get(arr):
            return arr[key]
        return cls._compose(get, obj)

    @classmethod
    def array(cls, arr):
        # arr = np.array([rvar._cast(var) for var in arr])

        # def make_array(*args):
        #     return np.array(args)
        # return cls._compose(make_array, *arr)
        return cls(arr)
