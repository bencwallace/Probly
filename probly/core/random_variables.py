"""
The base class for all random variables.

Random variables are defined as nodes in a computational graph that obey arithmetical and
(in)dependence relations.
"""

import functools
import warnings

import numpy as np
import scipy.misc

from numpy.lib.mixins import NDArrayOperatorsMixin

from .._exceptions import ConditionError, ConvergenceWarning
from .nodes import Node


class RandomVariable(Node, NDArrayOperatorsMixin):
    """
    A random variable.
    """

    def __init__(self, op=None, *parents):
        self.shape = ()

        # Initialize memo
        self._current_seed = None
        self._current_val = None

        if op is None:
            super().__init__()
        else:
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
        return self._sampler(*args)

    def _sampler(self, seed):
        raise NotImplementedError("_sampler not defined")

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
        return np.asarray(self.parents).reshape(self.shape)

    def __getitem__(self, key):
        def op(array):
            return array[key]
        return RandomVariable(op, self)

    # ------------------------------ Integrals ------------------------------ #

    def adjusted_mean(self, max_iter=int(1e5), tol=1e-5, adjustment=0):
        max_small_change = 100

        total = 0
        avg = 0
        count = 0
        for i in range(1, adjustment + 1):
            total += self(i)
        for i in range(adjustment + 1, max_iter):
            total += self(i)

            new_avg = total / (i - adjustment)
            delta = abs(new_avg - avg)
            avg = new_avg
            if delta <= tol:
                count += 1
                if count >= max_small_change:
                    return avg

        warnings.warn('Failed to converge.', ConvergenceWarning)
        return avg

    def mean(self, *args, **kwargs):
        return self.adjusted_mean(*args, **kwargs)

    def variance(self, *args, **kwargs):
        rv = (self - self.mean(*args, **kwargs)) ** 2
        return rv.adjusted_mean(adjustment=1, *args, **kwargs)

    def cdf(self, x, *args, **kwargs):
        return (self <= x).mean(*args, **kwargs)


class IndependentCopy(RandomVariable):
    _generator = np.random.default_rng(0)

    def __init__(self, op=None, *parents):
        self._offset = self._generator.integers(2 ** 32)
        super().__init__(op, *parents)

    def __call__(self, seed=None):
        new_seed = (self._seed(seed) + self._offset) % self._max_seed
        return super().__call__(new_seed)


class Conditional(RandomVariable):
    _max_attempts = 100_000

    def __init__(self, rv, *conditions):
        self.rv = rv
        self.conditions = conditions
        super().__init__()

    def _sampler(self, seed=None):
        seed = self._seed(seed)

        attempts = 0
        while not all([rv(seed) for rv in self.conditions]):
            seed = (seed + 1) % self._max_seed
            attempts += 1
            if attempts > self._max_attempts:
                raise ConditionError("Failed to meet condition")

        return self.rv(seed)


def seed(seed=None):
    """
    Seeds the current Probly session.
    """
    IndependentCopy._generator = np.random.default_rng(seed)
