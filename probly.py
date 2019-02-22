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


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(*args):
        """
        The lifted function

        Args:
            `RV`s and constants (though works with anything that casts to `RV`)
        """

        def sampler(seed):
            seed = gen_seed(seed)
            rv_samples = [RV(var)(seed) for var in args]
            return f(*rv_samples)
        # The following implicitly adds a node `X` to the graph.
        X = RV(sampler)

        # However, the node's `method` attribute is initially `sampler` and
        # must be changed to `f`
        RV.graph.nodes[X]['method'] = f

        # Edges must be added before losing information in `args`
        edges = [(RV(var), X, {'index': i}) for i, var in enumerate(args)]
        RV.graph.add_edges_from(edges)

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
    A generic random variable.

    A very general kind of random variable. Only capable of producing random
    quantities (on being called), whose explicit distribution is not
    necessarily known. However, compatible with arithmetical operations, can be
    acted upon by functions (using Lift), and subscriptable.

    Arguments:
        obj (RV, rv_generic, callable, numerical, or array-like)
    """

    graph = nx.DiGraph()

    def __new__(cls, obj):
        if isinstance(obj, cls):
            # "Copy" constructor. Should not be called by user.
            return obj
        else:
            return super().__new__(cls)

    def __init__(self, obj):
        if isinstance(obj, type(self)):
            # "Copy" constructor used (no need to init)
            return

        if isinstance(obj, rv_generic):
            # Initialize from scipy.stats random variable or similar
            self._sampler = lambda seed=None: obj.rvs(random_state=seed)

            RV.graph.add_node(self, method=self._sampler)
        elif callable(obj):
            # Initialization from sampler function (includes composition)

            try:
                obj(seed=0)
            except TypeError:
                # Case 1. Function composition
                # Case 2. Other sampler (for example from `random` or numpy)
                def sampler(seed=None):
                    random.seed(seed)
                    np.random.seed(seed)
                    return obj()
                self._sampler = sampler
            else:
                # Other/custom sampler
                self._sampler = lambda seed=None: obj(seed=seed)
        elif isinstance(obj, numbers.Number):
            # Constant (simplifies doing arithmetic)
            self._sampler = lambda seed=None: obj
        elif hasattr(obj, '__getitem__'):
            # Initialize from array-like (of dtype number, RV, array, etc.)
            array = np.array([RV(var) for var in obj])

            def sampler(seed):
                seed = gen_seed(seed)
                samples = [rv(seed) for rv in array]
                return np.array(samples)
            self._sampler = sampler

            RV.graph.add_node(self, method=np.array)
            edges = [(RV(var), self, {'index': i})
                     for i, var in enumerate(obj)]
            RV.graph.add_edges_from(edges)
        else:
            print('Error')

        if self not in RV.graph.nodes:
            # Only possible situation is "Case 2" above
            RV.graph.add_node(self, method=self._sampler)

    def __call__(self, seed=None):
        """Draw a random sample."""

        # Seed as follows to ensure independence of RVs with same sampler
        seed = gen_seed(seed)

        parents = list(self.parents())

        if len(list(parents)) == 0:
            return self._sampler((seed + id(self)) % _max_seed)
        else:
            samples = [p(seed) for p in parents]

            # Re-order parents according to edge index
            data = [RV.graph.get_edge_data(p, self) for p in parents]
            indices = [d['index'] for d in data]
            parents = [parents[i] for i in indices]

            # Sample from parents and evaluate method on samples
            method = RV.graph.nodes[self]['method']
            try:
                return method(samples)
            except TypeError:
                return method(*samples)

    def __getitem__(self, key):
        """Subscript a random matrix."""
        try:
            self()[0]
        except TypeError:
            raise TypeError('Scalar random variable is not subscriptable')

        def sampler(seed=None):
            sample = self(seed)
            return sample[key]
        X = RV(sampler)

        # Change implicit label
        select = lambda obj: op.__getitem__(obj, key)
        RV.graph.nodes[X]['method'] = select
        RV.graph.add_edge(self, X, index=0)

        return X

    # Define operators for emulating numeric types
    for p in _programs_lift + _programs_other + _programs_unary:
        exec(p)

    def parents(self):
        return list(RV.graph.predecessors(self))
