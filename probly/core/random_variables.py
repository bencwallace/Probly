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

from .._exceptions import ConditionError, ConvergenceWarning
from .nodes import Node


class RandomVariable(Node, NDArrayOperatorsMixin):
    """
    A random variable.
    """

    def __init__(self, op=None, *parents):
        # Scalar by default but overwritten by helpers.array
        self.shape = ()

        # Initialize memo
        self._current_seed = None
        self._current_val = None

        if op is None:
            op = self._sampler
        super().__init__(op, *parents)

    def copy(self):
        """Returns an independent, identically distributed random variable."""

        return IndependentCopy(self)

    def given(self, *conditions):
        """
        Returns a conditional random variable.

        :param conditions: RandomVariable
            Random variables with boolean samples.
        """
        return Conditional(self, *conditions)

    # ------------------------------ Sampling ------------------------------ #

    # NumPy max seed
    _max_seed = 2 ** 32 - 1

    def __call__(self, seed=None):
        """
        Returns a random sample of the random variable.
        """

        seed = self._seed(seed)

        # Check memo
        if seed == self._current_seed:
            return self._current_val
        else:
            # Recursively compute new value and update memo
            self._current_seed = seed
            self._current_val = super().__call__(self._current_seed)
            return self._current_val

    @classmethod
    def _seed(cls, seed=None):
        if seed is not None:
            return seed
        np.random.seed(seed)
        return np.random.randint(cls._max_seed)

    def _default_op(self, *args):
        return self._sampler(self, *args)

    def _sampler(self, seed):
        raise NotImplementedError

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

    def pdf(self, x, dx=1e-5, **kwargs):
        """Numerically approximates the pdf at x."""

        cdf = functools.partial(self.cdf, **kwargs)
        return scipy.misc.derivative(cdf, x, dx)


class IndependentCopy(RandomVariable):
    # Counter for _id. Set start=1 or else first RandomVariable acts as increment
    _current_id = itertools.count(start=1)

    def __init__(self, op=None, *parents):
        # Add _id and _offset attributes for independence
        self._id = next(self._current_id)
        np.random.seed(self._id)
        self._offset = np.random.randint(self._max_seed)
        super().__init__(op, *parents)

    def __call__(self, seed=None):
        new_seed = (self._seed(seed) + self._offset) % self._max_seed
        return super().__call__(new_seed)


class Conditional(RandomVariable):
    _max_attempts = 100_000

    def __init__(self, rv, *conditions):
        self.rv = rv
        self.conditions = conditions
        super().__init__(self._sampler)

    def _sampler(self, seed=None):
        seed = self._seed(seed)

        attempts = 0
        while not all([rv(seed) for rv in self.conditions]):
            seed = (seed + 1) % self._max_seed
            attempts += 1
            if attempts > self._max_attempts:
                raise ConditionError("Failed to meet condition")

        return self.rv(seed)
