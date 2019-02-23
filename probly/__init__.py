"""probly.py: A python module for working with random variables."""

import copy
import networkx as nx
import numpy as np
import operator as op
import random

# Local
from .programs import _programs
# from .helpers import Lift, get_seed

# For helpers
import math
from functools import wraps
from os import urandom


# Numpy max seed
_max_seed = 2 ** 32 - 1


# TO DO: Move to helpers.py
def Lift(f):
    """Lifts a function to the composition map between random variables."""

    @wraps(f)
    def F(*args):
        """
        The lifted function

        Args:
            `rvar`s and constants
        """

        return rvar._compose(f, *args)

    return F


def get_seed(seed=None):
    """
    Generate a random seed. If a seed is provided, returns it unchanged.

    Based on the Python implementation. A consistent approach to generating
    re-usable random seeds is needed in order to implement dependency.
    """

    if seed is not None:
        return seed

    try:
        max_bytes = math.ceil(np.log2(_max_seed) / 8)
        seed = int.from_bytes(urandom(max_bytes), 'big')
    except NotImplementedError:
        raise

    return seed


class rvarGen(object):
    """
    A general random variable.

    Can be sampled from and acted upon by appropriate functions decorated with
    `Lift`.
    """

    graph = nx.MultiDiGraph()

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
        seed = get_seed(seed)
        parents = list(self.parents())

        if len(parents) == 0:
            # Seed as follows for independence of `rvarGen`s with same `sampler`
            return self.sampler((seed + id(self)) % _max_seed)
        else:
            # Create {index: parent} dictionary `arguments`
            data = [rvarGen.graph.get_edge_data(p, self) for p in parents]
            arguments = {}
            for i in range(len(parents)):
                indices = [d.values() for d in data[i].values()]
                for j in range(len(indices)):
                    arguments[data[i][j]['index']] = parents[i]

            # Sample elements of `parents` in order specified by `arguments`
            # and apply `method` to result
            samples = [arguments[i](seed) for i in range(len(arguments))]
            method = rvarGen.graph.nodes[self]['method']
            return method(*samples)

    def parents(self):
        """Returns list of random variables from which `self` is defined"""

        if self in rvarGen.graph:
            return list(rvarGen.graph.predecessors(self))
        else:
            return []

    # Constructors
    @classmethod
    def _compose(cls, f, *args):
        composed_rvar = cls.__new__(cls)

        rvar.graph.add_node(composed_rvar, method=f)
        edges = [(rvar._cast(var), composed_rvar, {'index': i})
                 for i, var in enumerate(args)]
        rvar.graph.add_edges_from(edges)

        return composed_rvar

        rvarGen.graph.add_node(composed_rvar, method=f)
        edges = [(rvarGen._cast(var), composed_rvar, {'index': i})
                 for i, var in enumerate(args)]
        rvarGen.graph.add_edges_from(edges)

        return composed_rvar

    @classmethod
    def _cast(cls, obj):
        """Cast constants to `rvarGen` objects."""

        if isinstance(obj, cls):
            return obj
        else:
            return cls(lambda seed=None: obj)

    def copy(self, obj):
        """Return a random variable with the same distribution as `self`"""

        # Shallow copy is ok as `rvarGen` isn't mutable
        return copy.copy(self)

    # @classmethod
    # def define(cls, sampler, origin='random'):
    #     """Define a random variable from a pre-defined sampler"""

    #     if origin is None:
    #         return cls(sampler)
    #     # Following are obsolete for this class
    #     elif origin == 'scipy':
    #         # Note: scipy uses np.random.seed
    #         return cls(lambda seed=None: sampler.rvs(random_state=seed))
    #     elif origin == 'numpy':
    #         def seeded_sampler(seed=None):
    #             np.random.seed(seed)
    #             return sampler()
    #         return cls(seeded_sampler)
    #     elif origin == 'random':
    #         def seeded_sampler(seed=None):
    #             random.seed(seed)
    #             return sampler()
    #         return cls(seeded_sampler)
    #     else:
    #         raise TypeError('Unknown origin `{}`'.format(origin))


class rvarNumeric(rvarGen):
    """
    A random variable of numeric type. Not for direct use.

    Compatible with numerical operations.
    """

    # Define operators for emulating numeric types
    for p in _programs:
        exec(p)


class rvar(rvarNumeric):
    """A random scalar."""

    pass


class rarray(rvarNumeric):
    """
    A random array.

    Supports subscripting and matrix operations.
    """

    def __new__(cls, arr):
        arr = np.array([rvar._cast(var) for var in arr])

        def make_array(*args):
            return np.array(args)

        return cls._compose(make_array, *arr)

    def __init__(self, arr):
        pass

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
