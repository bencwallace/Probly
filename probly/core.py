"""
This module defines a random variable as a node in a computational graph.
"""

import itertools
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class Node(object):
    """
    A node in a computational graph.

    Parameters
    ----------
    call_method : callable
    parents : list
        List of `Node` objects

    Note
    ----
    `Node` objects are not meant to be instantiated directly.
    """

    # Core magic methods
    def __init__(self, op=None, *parents):
        """
        Parameters
        ----------
        op : optional
            An operation. Default is the identity map. If not callable,
            understood as the constant map returning itself. Mandatory if
            parents not specified.
        parents : optional
            A collection of callables (preferably `SimpleNode` objects).
        """

        self._parents = parents

        if not op:
            # Must be root acting as identity
            assert not parents
            self._op = lambda *x: x
        elif not callable(op):
            # Treat op as constant
            self._op = lambda *x: op
        else:
            self._op = op

    def __call__(self, *args):
        """
        Combines parent outputs on given arguments via an operation.
        """

        if not self._parents:
            # Let root act directly on args
            out = self._op(*args)
        else:
            inputs = (p(*args) for p in self._parents)
            out = self._op(*inputs)

        # For length 1 tuples
        if hasattr(out, '__len__') and len(out) == 1:
            out = out[0]

        return out


class RandomVar(Node, NDArrayOperatorsMixin):
    """
    A random variable.

    Behaves like a `Node` object that can be acted on by any NumPy ufunc in a
    way compatible with the graph structure. Also acts as an interface for the
    definition of families of random variables via subclassing. Such families
    should be defined by subclassing RandomVar rather than Node.

    Example
    -------
    Define a family of "shifted" uniform random variables:

    >>> import numpy as np
    >>> import probly as pr
    >>> class UnifShift(pr.Distr):
    ...     def __init__(self, a, b):
    ...         self.a = a + 1
    ...         self.b = b + 1
    ...     def _sampler(self, seed=None):
    ...         np.random.seed(seed)
    ...         return np.random.uniform(self.a, self.b)

    Instantiate a random variable from this family with support `[1, 2]` and
    sample from its distribution:

    >>> X = UnifShift(0, 1)
    >>> X()
    """

    # Counter for _id. Set start=1 or else first RandomVar acts as increment
    _last_id = itertools.count(start=1)

    # NumPy max seed
    _max_seed = 2 ** 32 - 1

    @staticmethod
    def get_seed(seed=None, offset=False):
        """
        Produces a random seed.
        """

        np.random.seed(seed)
        new_seed = np.random.get_state()[1][int(offset)]

        # Cast to int to avoid overflow
        return int(new_seed)

    def _sampler(seed=None):
        # Defaults to identity map.
        # Due to _offset, behaves as random number generator
        return seed

    def __new__(cls, *args, **kwargs):
        """
        Defines the subclassing interface.

        A subclass of RandomVar contains: a method _sampler that produces
        samples from a distribution given some seed; and possibly a method
        __init__ to specify parameters. The __new__ method initializes an
        object of such a subclass as a Node object with no parents and with
        _op given by the _sampler method. In addition, __new__ adds _id and
        _offset attributes that are used to ensure independence.
        """

        # First constructs a bare Node object
        obj = super().__new__(cls)

        if cls is RandomVar:
            # Initialize from arguments
            op = args[0]
            parents = args[1:]
        else:
            # Initialize from _sampler
            op = obj._sampler
            parents = ()

        # Then initialize it
        super().__init__(obj, op, *parents)

        # Add _id and _offset attributes for independence.
        obj._id = next(cls._last_id)
        obj._offset = cls.get_seed(obj._id, offset=True)

        return obj

    # This method allows for compatibility NumPy ufuncs.
    def __array_ufunc__(self, op, method, *parents, **kwargs):
        # Cast parents to RandomVar: If not RandomVar, treat as constant
        parents = tuple(p if isinstance(p, RandomVar)
                        else RandomVar(p) for p in parents)

        def partial(*parents):
            return getattr(op, method)(*parents, **kwargs)

        return RandomVar(partial, *parents)

    def __call__(self, seed=None):
        if self._parents:
            # Random variable depends on parents
            return super().__call__(seed)
        else:
            # Independent random variable
            new_seed = (self.get_seed(seed) + self._offset) % self._max_seed
            return super().__call__(new_seed)

    def __getitem__(self, key):
        def get_item_from_key(array):
            return array[key]
        return RandomVar(get_item_from_key, self)
