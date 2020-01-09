"""
The base class for all random variables.

Random variables are defined as nodes in a computational graph that obey arithmetical and
(in)dependence relations.
"""

import functools
import itertools
import warnings

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
import scipy.misc

from .exceptions import ConditionError, ConvergenceWarning


class Node(object):
    def __init__(self, op, *parents):
        self.parents = parents

        if not callable(op):
            # Treat op as constant
            self.op = lambda *x: op
        else:
            self.op = op

    def __call__(self, *args):
        if not self.parents:
            # Let root act directly on args
            out = self.op(*args)
        else:
            inputs = (p(*args) for p in self.parents)
            out = self.op(*inputs)

        # For length 1 tuples
        if hasattr(out, '__len__') and len(out) == 1:
            out = out[0]

        return out


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

    def copy(self):
        """Returns an independent, identically distributed random variable."""

        return IndependentCopy(self)

    # ----------------------------- Constructor ----------------------------- #

    def __init__(self, op, *parents):
        # Scalar by default but overwritten by helpers.array
        self.shape = ()

        # Initialize memo
        self._current_seed = None
        self._current_val = None

        super().__init__(op, *parents)

    # ------------------------------ Sampling ------------------------------ #

    # NumPy max seed
    _max_seed = 2 ** 32 - 1

    def __call__(self, seed=None):
        """
        Produces a random sample.
        """

        seed = self._get_seed(seed)

        # Check memo
        if seed == self._current_seed:
            return self._current_val
        else:
            # Recursively compute new value and update memo
            self._current_seed = self._get_seed(seed)
            self._current_val = super().__call__(self._current_seed)
            return self._current_val

    @classmethod
    def _get_seed(cls, seed=None, return_if_seeded=True):
        """
        Produces a random seed.

        If `return_if_seeded` is True, returns `seed` (when provided).
        """

        if seed is not None and return_if_seeded:
            return seed

        np.random.seed(seed)
        return np.random.randint(cls._max_seed)

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
            return np.asarray(self.parents).reshape(self.shape)
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

    # todo: eq and hash
    def __hash__(self):
        return id(self)


class RandomVariableWithIndependence(RandomVariable):
    # Counter for _id. Set start=1 or else first RandomVariable acts as increment
    _current_id = itertools.count(start=1)

    def __init__(self, op, *parents):
        # Add _id and _offset attributes for independence
        self._id = next(self._current_id)
        self._offset = self._get_random(self._id)
        super().__init__(op, *parents)

    def __call__(self, seed=None):
        new_seed = (self._get_seed(seed) + self._offset) % self._max_seed
        return super().__call__(new_seed)

    @classmethod
    def _get_random(cls, seed):
        """
        Produces a pseudo-random number from a given input seed.
        """

        return cls._get_seed(seed, False)


class IndependentCopy(RandomVariableWithIndependence):
    def __init__(self, rv):
        super().__init__(rv.op, *rv.parents)


class Distribution(RandomVariableWithIndependence):
    def __init__(self):
        op = self._sampler
        super().__init__(op)


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
