"""probly.py: A python module for working with random variables."""

import math
import networkx as nx
import numpy as np
import operator as op

import random

from os import urandom

# Numpy max seed
_max_seed = 2 ** 32 - 1

# Exec programs for automating repetitive operator definitions
_num_ops_lift = ['add', 'sub', 'mul', 'matmul',
                 'truediv', 'floordiv', 'mod', 'divmod', 'pow']
_num_ops_right = ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod',
                  'divmod', 'pow']
_num_ops_unary = ['neg', 'pos', 'abs', 'complex', 'int', 'float', 'round',
                  'trunc', 'floor', 'ceil']

_programs_lift = [
    (
        'def __{:s}__(self, x):\n'
        '    return Lift(op.{:s})(self, x)'
    ).format(fcn, fcn) for fcn in _num_ops_lift]

_programs_right = [
    (
        'def __r{:s}__(self, x):\n'
        '   X = RV.make_rv(x)\n'
        '   return X.__{:s}__(self)'
    ).format(fcn, fcn) for fcn in _num_ops_right]

_programs_unary = [
    (
        'def __{:s}__(self):\n'
        '   return Lift(op.{:s})(self)'
    ).format(fcn, fcn) for fcn in _num_ops_unary]
_programs = _programs_lift + _programs_right + _programs_unary


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(*args):
        """
        The lifted function

        Args:
            `RV`s and constants (though works with anything that casts to `RV`)
        """

        return RV.compose(f, *args)

    return F


def gen_seed(seed=None):
    """
    Generate a random seed. If a seed is provided, returns it unchanged.

    Based on the Python implementation. A consistent approach to generating
    re-usable random seeds is needed in order to implement dependency.

    Note: numpy requires seeds between 0 and 2 ** 32 - 1.
    """

    if seed is not None:
        return seed

    try:
        max_bytes = math.ceil(np.log2(_max_seed) / 8)
        seed = int.from_bytes(urandom(max_bytes), 'big')
    except NotImplementedError:
        raise

    return seed


class RV(object):
    """
    A random variable placeholder.

    Can be acted upon by arithmetical operations and functions compatible with
    `Lift`.
    """

    graph = nx.DiGraph()

    def __init__(self, sampler=None, f=None, *args):
        self.sampler = sampler
        if self.sampler is not None:
            assert callable(sampler), '`sampler` is not callable'

        if f is not None:
            RV.graph.add_node(self, method=f)
            edges = [(RV.make_rv(var), self, {'index': i})
                     for i, var in enumerate(args)]
            RV.graph.add_edges_from(edges)

    def __call__(self, seed=None):
        seed = gen_seed(seed)
        parents = list(self.parents())

        if len(parents) == 0:
            return self.sampler(seed)
        else:
            # Re-order parents according to edge index
            data = [RV.graph.get_edge_data(p, self) for p in parents]
            indices = [d['index'] for d in data]
            parents = [parents[i] for i in indices]

            # Sample from parents and evaluate method on samples
            samples = [p(seed) for p in parents]
            method = RV.graph.nodes[self]['method']
            return method(*samples)

    # Define operators for emulating numeric types
    for p in _programs:
        exec(p)

    def parents(self):
        if self in RV.graph:
            return list(RV.graph.predecessors(self))
        else:
            return []

    @classmethod
    def make_rv(cls, obj):
        """Make RV from constant or RV"""
        if isinstance(obj, cls):
            return obj
        else:
            return cls(lambda seed=None: obj)

    @classmethod
    def from_random(cls, random_sampler):
        def seeded_sampler(seed=None):
            random.seed(seed)
            return random_sampler()
        return cls(seeded_sampler)

    @classmethod
    def from_numpy(cls, numpy_sampler):
        def seeded_sampler(seed=None):
            np.random.seed(seed)
            return numpy_sampler()
        return cls(seeded_sampler)

    @classmethod
    def from_scipy(cls, scipy_rv):
        return cls(lambda seed=None: scipy_rv.rvs(random_state=seed))

    @classmethod
    def compose(cls, f, *args):
        return cls(None, f, *args)

    @classmethod
    def array(cls, arr):
        def make_array(*args):
            return np.array(args)
        return cls(None, make_array, *arr)
