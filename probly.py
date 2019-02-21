"""probly.py: A python module for working with random variables."""

import numbers
import numpy as np
import operator as op

from os import urandom
import time

from warnings import warn

# For automating repetitive operator definitions
num_ops_lift = ['add', 'sub', 'mul', 'matmul',
                'truediv', 'floordiv', 'mod', 'divmod', 'pow']
num_ops_right = ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod',
                 'divmod', 'pow']
num_ops_unary = ['neg', 'pos', 'abs', 'complex', 'int', 'float', 'round',
                 'trunc', 'floor', 'ceil']

programs_lift = [
    (
        'def __{:s}__(self, x):\n'
        '    return Lift(op.{:s})(self, x)'
    ).format(fcn, fcn) for fcn in num_ops_lift]

programs_other = [
    (
        'def __r{:s}__(self, x):\n'
        '   X = RV(x)\n'
        '   return X.__{:s}__(self)'
    ).format(fcn, fcn) for fcn in num_ops_right]

programs_unary = [
    (
        'def __{:s}__(self):\n'
        '   return Lift(op.{:s})(self)'
    ).format(fcn, fcn) for fcn in num_ops_unary]


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(*args):
        def sampler(seed):
            seed = gen_seed(seed)
            rv_samples = [rv(seed) for rv in args]
            return f(*rv_samples)
        return RV(sampler)

    return F


def gen_seed(seed=None):
    """
    Generate a random seed.

    Based on Python's implementation. Note: numpy requires seeds between 0 and
    2 ** 32 - 1.
    """

    if seed is not None:
        return seed

    try:
        seed = int.from_bytes(urandom(4), 'big')
    except NotImplementedError:
        print('Need to implement method to seed from time')

    return seed


class RV(object):
    """
    A generic random RV.

    Basically a very general kind of random variable. Only capable of producing
    random quantities, whose explicit distribution is not necessarily known.

    Attributes:
        species (str)
        sample (function)
    """

    def __new__(cls, obj):
        if isinstance(obj, cls):
            # "Copy" constructor. Should not be called by user.
            return obj
        else:
            return super().__new__(cls)

    def __init__(self, obj):
        """Type conversion."""

        if isinstance(obj, type(self)):
            # "Copy" constructor --> No need to init
            pass
        elif hasattr(obj, 'rvs'):
            # Initialize from scipy.stats random variable or similar
            self.species = 'scipy'
            self._sampler = lambda seed=None: obj.rvs(random_state=seed)
        elif callable(obj):
            # Direct initialization from `_sampler` function
            self.species = 'custom'

            # Determine whether to use parameter name `seed` or `random_state`
            try:
                obj(seed=0)
            except TypeError:
                fail = True
            else:
                fail = False
                self._sampler = lambda seed: obj(seed=seed)
                return

            try:
                obj(random_state=0)
            except TypeError:
                fail = True
            else:
                fail = False
                self._sampler = lambda seed: obj(random_state=seed)
                return

            if fail:
                self._sampler = lambda seed: obj(seed)
                warn('Sampler function seeding argument not found')
        elif isinstance(obj, numbers.Number):
            # Constant (simplifies doing arithmetic)
            self.species = 'const'
            self._sampler = lambda seed=None: obj
        elif hasattr(obj, '__getitem__'):
            # Initialize from array-like (of dtype number, RV, array, etc.)
            self.species = 'array'
            array = np.array([RV(item) for item in obj])

            def sampler(seed):
                seed = gen_seed(seed)
                samples = [rv(seed) for rv in array]
                return np.array(samples)
            self._sampler = sampler
        else:
            print('Error')

    def __call__(self, seed=None):
        """Draw a random sample."""

        return self._sampler(seed)

    def __getitem__(self, key):
        try:
            self()[0]
        except TypeError:
            raise TypeError('Scalar random variable is not subscriptable')

        if True:
            def sampler(seed=None):
                sample = self(seed)
                return sample[key]
            return RV(sampler)
        else:
            return RV(self.arg[key])

    # Operators for emulating numeric types
    for p in programs_lift + programs_other + programs_unary:
        exec(p)
