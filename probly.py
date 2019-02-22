"""probly.py: A python module for working with random variables."""

import math
import networkx as nx
import numbers
import numpy as np
import operator as op

import random
from scipy.stats._distn_infrastructure import rv_generic

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

_programs_other = [
    (
        'def __r{:s}__(self, x):\n'
        '   X = RV(x)\n'
        '   return X.__{:s}__(self)'
    ).format(fcn, fcn) for fcn in _num_ops_right]

_programs_unary = [
    (
        'def __{:s}__(self):\n'
        '   return Lift(op.{:s})(self)'
    ).format(fcn, fcn) for fcn in _num_ops_unary]
_programs = _programs_lift + _programs_other + _programs_unary

def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(*args):
        """
        The lifted function

        Args:
            `RV`s and constants (though works with anything that casts to `RV`)
        """

        # The following implicitly adds a node `X` to the graph.
        X = rvCompound(f, *args)

        return X

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
        print('Need to implement method to seed from time')

    return seed


class RV(object):
    """
    A random variable placeholder.

    Can be acted upon by arithmetical operations and functions compatible with
    Lift.
    """

    graph = nx.DiGraph()

    # Define operators for emulating numeric types
    for p in _programs:
        exec(p)


class Const(RV):
    def __init__(self, value):
        self.value = value

    def __call__(self, seed=None):
        return self.value


class rvIndep(RV):
    def __init__(self, sampler):
        if isinstance(sampler, rv_generic):
            # Initialize from scipy.stats random variable or similar
            self._sampler = lambda seed=None: sampler.rvs(random_state=seed)
        elif callable(sampler):
            # Initialization from sampler function

            try:
                # Check if `sampler` takes `seed` as an argument
                sampler(seed=0)
            except TypeError:
                # Case 1. No `seed` argument (assume `random` or `numpy`)
                def seeded_sampler(seed=None):
                    random.seed(seed)
                    np.random.seed(seed)
                    return sampler()
                self._sampler = seeded_sampler
            else:
                # Case 2. Sampler with `seed` argument
                self._sampler = lambda seed=None: sampler(seed=seed)

    def __call__(self, seed=None):
        seed = gen_seed(seed)
        return self._sampler((seed + id(self)) % _max_seed)


class rvCompound(RV):
    def __init__(self, f, *args):
        RV.graph.add_node(self, method=f)

        edges = [(rv_cast(var), self, {'index': i})
                 for i, var in enumerate(args)]
        RV.graph.add_edges_from(edges)

    def __call__(self, seed=None):
        """Draw a random sample."""

        # Seed as follows to ensure independence of RVs with same sampler
        seed = gen_seed(seed)

        parents = list(self.parents())
        samples = [p(seed) for p in parents]

        # Re-order parents according to edge index
        data = [RV.graph.get_edge_data(p, self) for p in parents]
        indices = [d['index'] for d in data]
        parents = [parents[i] for i in indices]

        # Sample from parents and evaluate method on samples
        method = RV.graph.nodes[self]['method']
        return method(*samples)

    def parents(self):
        return list(RV.graph.predecessors(self))


class rvArray(rvCompound):
    def __init__(self, array):
        super().__init__(self, np.array, *array)

    def __getitem__(self, key):
        """Subscript a random matrix."""

        # assert hasattr(self(), '__getitem__'), 'Scalar random variable is\
        #                                         not subscriptable'

        get = lambda *args: op.__getitem__([args], key)
        X = Lift(get)(self)

        # Change implicit label (shouldn't be needed anymore)
        # select = lambda *args: op.__getitem__([args], key)
        # RV.graph.nodes[X]['method'] = select
        RV.graph.add_edge(self, X, index=0)

        return X


def rvarray(array):
    # X = RV()
    # RV.graph.add_node(X, method=lambda *args: np.array(args))
    # edges = [(rv_cast(var), X, {'index': i})
    #          for i, var in enumerate(array)]
    # RV.graph.add_edges_from(edges)
    # return X
    return Lift(np.array)(array)


def rv_cast(var):
    """Cast constants to RVs"""
    if isinstance(var, RV):
        return var
    else:
        return Const(var)
