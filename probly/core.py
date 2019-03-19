"""
Random variables are implemented as a kind of node in a computational graph.

The basic type underlying all random variables is a node in a computational
graph. Random variables are defined as nodes that obey arithmetical and
independence relations.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import scipy.misc

import itertools
import functools
import copy

import warnings
from .exceptions import ConvergenceWarning


class Node(object):
    """
    A node in a computational graph, representing the output of an operation.

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

        if op is None:
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

    def copy(self):
        """Returns a copy of the current node."""

        return copy.copy(self)

    # Getters
    def op(self):
        """Returns the operation producing the value of the current node."""

        return self._op

    def parents(self):
        """Returns the parents of the current node."""
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

    >>> from probly.core import RandomVar
    >>> import numpy as np
    >>> class UnifShift(RandomVar):
    ...     def __init__(self, a, b):
    ...         self.a = a + 1
    ...         self.b = b + 1
    ...     def _sampler(self, seed=None):
    ...         np.random.seed(seed)
    ...         return np.random.uniform(self.a, self.b)

    Instantiate a random variable from this family with support `[1, 2].

    >>> X = UnifShift(0, 1)
    """

    # ---------------------------- Independence ---------------------------- #

    # Counter for _id. Set start=1 or else first RandomVar acts as increment
    _last_id = itertools.count(start=1)

    @classmethod
    def _reset(cls):
        # For debugging
        cls._last_id = itertools.count(start=1)

    def copy(self):
        """Returns an independent, identically distributed random variable."""

        Copy = super().copy()

        Copy._id = next(self._last_id)
        Copy._offset = self._get_random(Copy._id)

        return Copy

    # ----------------------------- Constructor ----------------------------- #

    def __new__(cls, *args, **kwargs):
        """
        Defines the random variable subclassing interface.

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

            _offset = cls._get_random(_id)

        # Then initialize it
        super().__init__(obj, op, *parents)

        # Add _id and _offset attributes for independence
        obj._id = _id
        obj._offset = _offset

        # Scalar by default but overwritten by helpers.array
        obj.shape = ()

        return obj

    # ------------------------------ Sampling ------------------------------ #

    # NumPy max seed
    _max_seed = 2 ** 32 - 1

    def __call__(self, seed=None):
        """
        Produces a random sample.
        """

        new_seed = (self._get_seed(seed) + self._offset) % self._max_seed
        return super().__call__(new_seed)

    @classmethod
    def _get_seed(cls, seed=None, return_if_seeded=True):
        """
        Produces a random seed.

        If `return_if_seeded` is True, returns `seed` (when provided).
        """

        if seed and return_if_seeded:
            return seed

        np.random.seed(seed)
        return np.random.randint(cls._max_seed)

    @classmethod
    def _get_random(cls, seed, old=False):
        """
        Produces a pseudo-random number from a given input seed.
        """

        return cls._get_seed(seed, False)

    def _sampler(self, seed=None):
        # Default sampler. Behaves as random number generator when called via
        # __call__, which adds _offset to seed.

        return self._get_seed(seed)

    # ------------------------ Arrays and arithmetic ------------------------ #

    def __array_ufunc__(self, op, method, *inputs, **kwargs):
        # Allows NumPy ufuncs (e.g. addition) and derived methods
        # (e.g. summation, which is the ufunc reduce method of addition)
        # to act on RandomVar objects.

        # Cast inputs to RandomVar: If not RandomVar, treat as constant
        inputs = tuple(x if isinstance(x, RandomVar)
                       else RandomVar(x) for x in inputs)

        fcn = getattr(op, method)
        partial = functools.partial(fcn, **kwargs)

        return RandomVar(partial, *inputs)

    def __array__(self, dtype=object):
        # Determines behaviour of np.array
        if self.shape:
            # Return the array represented by self
            return np.asarray(self.parents()).reshape(self.shape)
        else:
            # Form a single-element array
            arr = np.ndarray(1, dtype=object)
            arr[0] = self
            return arr

    def __getitem__(self, key):
        if not self.shape:
            raise TypeError('Scalar random variable is not subscriptable.')

        def get_item_from_key(array):
            return array[key]
        return RandomVar(get_item_from_key, self)

    # ------------------------------ Integrals ------------------------------ #

    # All integrals (including probabilities) are defined in terms of the mean
    def mean(self, max_iter=int(1e5), tol=1e-5):
        """Numerically approximates the mean."""

        max_small_change = 100

        total = 0
        avg = 0
        count = 0
        delta = tol + 1
        for i in range(1, max_iter):
            total += self(i)
            new_avg = total / i
            delta = abs(new_avg - avg)
            avg = new_avg

            if delta <= tol:
                count += 1
                if count >= max_small_change:
                    return avg

        warnings.warn('Failed to converge.', ConvergenceWarning)
        return avg

    def moment(self, p, **kwargs):
        """Numerically approximates the p-th moment."""

        rv = self ** p
        return rv.mean(**kwargs)

    def cmoment(self, p, **kwargs):
        """Numerically approximates the p-th central moment."""

        # To do: Subclasses must allow kwargs
        return self.moment(p, **kwargs) - (self.mean()) ** p

    def Var(self, **kwargs):
        """Numerically approximates the variance."""

        return self.cmoment(2, **kwargs)

    def cdf(self, x, **kwargs):
        """Numerically approximates the cdf at x."""

        return (self <= x).mean(**kwargs)

    # Quite slow and inaccurate
    def pdf(self, x, dx=1e-5, **kwargs):
        """Numerically approximates the pdf at x."""

        cdf = functools.partial(self.cdf, **kwargs)
        return scipy.misc.derivative(cdf, x, dx)
