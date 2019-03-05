"""
This module defines a random variable as a node in a computational graph.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import itertools
import functools


class Node(object):
    """
    A node in a computational graph.

    Acts as a function by passing the outputs of its `parents` (computed
    recursively) to an operation `op`.

    Parameters
    ----------
    op : optional
        An operation accepting `len(parents)` arguments.
        Defaults to the identity map.
        If not callable, understood as the constant map returning itself.
        Mandatory if parents not specified.
    parents : optional
        A collection of callable (typically `Node`) objects.
    """

    def __init__(self, op=None, *parents):
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
        if not self.parents():
            # Let root act directly on args
            out = self.op()(*args)
        else:
            inputs = (p(*args) for p in self.parents())
            out = self.op()(*inputs)

        # For length 1 tuples
        if hasattr(out, '__len__') and len(out) == 1:
            out = out[0]

        return out

    # Getters
    def op(self):
        return self._op

    def parents(self):
        return self._parents


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
    >>> class UnifShift(RandomVar):
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

        _id = next(cls._last_id)

        # First constructs a bare Node object
        obj = super().__new__(cls)

        if cls is RandomVar:
            # Dependent RandomVar initialized from args
            op = args[0]
            parents = args[1:]

            _offset = 0
        else:
            # Independent RandomVar initialized from _sampler
            op = obj._sampler
            parents = ()

            _offset = cls.get_random(_id)

        # Then initialize it
        super().__init__(obj, op, *parents)

        # Add _id and _offset attributes for independence.
        obj._id = _id
        obj._offset = _offset

        return obj

    def __call__(self, seed=None):
        """
        Produces a random sample.
        """

        new_seed = (self.get_seed(seed) + self._offset) % self._max_seed
        return super().__call__(new_seed)

    @classmethod
    def get_seed(cls, seed=None, return_if_seeded=True):
        """
        Produces a random seed.

        If `return_if_seeded` is True, returns `seed` (when provided).
        """

        if seed and return_if_seeded:
            return seed

        np.random.seed(seed)
        return np.random.randint(cls._max_seed)

    @classmethod
    def get_random(cls, seed, old=False):
        """
        Produces a pseudo-random number from a given input seed.
        """

        return cls.get_seed(seed, False)

    def __array_ufunc__(self, op, method, *inputs, **kwargs):
        # Allows NumPy ufuncs (in particular, addition) and derived methods
        # (for example, summation, which is the ufunc reduce method of
        # addition) to act on RandomVar objects.

        # Cast inputs to RandomVar: If not RandomVar, treat as constant
        inputs = tuple(x if isinstance(x, RandomVar)
                       else RandomVar(x) for x in inputs)

        fcn = getattr(op, method)
        partial = functools.partial(fcn, **kwargs)

        return RandomVar(partial, *inputs)

    def __array__(self, dtype=object):
        # Determines behaviour of np.array
        if hasattr(self, '_array'):
            items = [p.__array__() for p in self.parents()]
            return np.array(items, dtype=object)
        else:
            return self

    def __getitem__(self, key):
        def get_item_from_key(array):
            return array[key]
        return RandomVar(get_item_from_key, self)

    def _sampler(self, seed=None):
        # Default sampler. Behaves as random number since called through
        # __call__, which adds _offset to seed.

        return self.get_seed(seed)
