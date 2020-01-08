"""
The base class for all random variables.

Random variables are defined as nodes in a computational graph that obey arithmetical and
(in)dependence relations.
"""

import copy
import functools
import itertools
import warnings

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import scipy.misc

from .exceptions import ConditionError, ConvergenceWarning


class Node(object):
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


class RandomVariable(Node, NDArrayOperatorsMixin):
    """
    A random variable.

    Behaves like a `Node` object that can be acted on by any NumPy ufunc in a
    way compatible with the graph structure. Also acts as an interface for the
    definition of families of random variables via subclassing. Such families
    should be defined by subclassing RandomVariable rather than Node.

    Example
    -------
    Define a family of "shifted" uniform random variables:

    >>> from probly.core import RandomVariable
    >>> import numpy as np
    >>> class UnifShift(RandomVariable):
    ...     def __init__(self, a, b):
    ...         self.a = a + 1
    ...         self.b = b + 1
    ...     def _sampler(self, seed=None):
    ...         np.random.seed(seed)
    ...         return np.random.uniform(self.a, self.b)

    Instantiate a random variable from this family with support `[1, 2]`.

    >>> X = UnifShift(0, 1)
    """

    # ---------------------------- Independence ---------------------------- #

    # Counter for _id. Set start=1 or else first RandomVariable acts as increment
    _current_id = itertools.count(start=1)

    def copy(self):
        """Returns an independent, identically distributed random variable."""

        c = super().copy()

        c._id = next(self._current_id)
        c._offset = self._get_random(c._id)

        return c

    # ----------------------------- Constructor ----------------------------- #

    def __new__(cls, *args, **kwargs):
        """
        Defines the random variable subclassing interface.

        A subclass of RandomVariable contains: a method _sampler that produces
        samples from a distribution given some seed; and possibly a method
        __init__ to specify parameters. The __new__ method initializes an
        object of such a subclass as a Node object with no parents and with
        _op given by the _sampler method. In addition, __new__ adds _id and
        _offset attributes that are used to ensure independence.
        """

        _id = next(cls._current_id)

        # First constructs a bare Node object
        obj = super().__new__(cls)

        if cls is RandomVariable:
            # Dependent RandomVariable initialized from args
            op = args[0]
            parents = args[1:]

            _offset = 0
        else:
            # Independent RandomVariable initialized from _sampler
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

        # Initialize memo
        obj.prev_seed = None
        obj.prev_val = None

        return obj

    # ------------------------------ Sampling ------------------------------ #

    # NumPy max seed
    _max_seed = 2 ** 32 - 1

    def __call__(self, seed=None):
        """
        Produces a random sample.
        """

        # Check memo
        if seed == self.prev_seed:
            return self.prev_val
        else:
            # Recursively compute new value and update memo
            self.prev_seed = (self._get_seed(seed) + self._offset) % self._max_seed
            self.prev_val = super().__call__(self.prev_seed)
            return self.prev_val

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
    def _get_random(cls, seed):
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
        # to act on RandomVariable objects.

        # Cast inputs to RandomVariable: If not RandomVariable, treat as constant
        inputs = tuple(x if isinstance(x, RandomVariable)
                       else RandomVariable(x) for x in inputs)

        fcn = getattr(op, method)
        partial = functools.partial(fcn, **kwargs)

        return RandomVariable(partial, *inputs)

    def __array__(self, dtype=object):
        # Determines behaviour of np.array
        if self.shape:
            # Return the array represented by self
            return np.asarray(self._parents).reshape(self.shape)
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
        return RandomVariable(get_item_from_key, self)


    # - Conditioning - #

    def given(self, condition):
        return Conditional(self, condition)

    # ------------------------------ Integrals ------------------------------ #

    # All integrals (including probabilities) are defined in terms of the mean
    def mean(self, max_iter=int(1e5), tol=1e-5):
        """Numerically approximates the mean."""

        max_small_change = 100

        total = 0
        avg = 0
        count = 0
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

        # todo: Subclasses must allow kwargs
        return self.moment(p, **kwargs) - (self.mean()) ** p

    def variance(self, **kwargs):
        """Numerically approximates the variance."""

        return self.cmoment(2, **kwargs)

    def cdf(self, x, **kwargs):
        """Numerically approximates the cdf at x."""

        return (self <= x).mean(**kwargs)

    # todo: slow and inaccurate
    def pdf(self, x, dx=1e-5, **kwargs):
        """Numerically approximates the pdf at x."""

        cdf = functools.partial(self.cdf, **kwargs)
        return scipy.misc.derivative(cdf, x, dx)

    # - Other - #

    def __hash__(self):
        return id(self)


class Conditional(RandomVariable):
    _max_attempts = 100_000

    def __init__(self, rv, condition):
        self.rv = rv
        self.condition = condition

    def __call__(self, seed=None):
        seed = self._get_seed(seed)

        attempts = 0
        while not self.condition.check(seed):
            seed = (seed + 1) % self._max_seed
            attempts += 1
            if attempts > self._max_attempts:
                raise ConditionError("Failed to meet condition")

        return self.rv(seed)
