"""probly.py: A python module for working with random variables."""

import numbers
import numpy as np
import operator as op
from os import urandom
import time

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
        def sample(seed):
            seed = gen_seed(seed)
            rv_samples = [rv(seed) for rv in args]
            return f(*rv_samples)
        return RV(sample)

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

    def __init__(self, obj):
        """Type conversion."""

        if callable(obj):
            # Direct initialization from sample function
            self.species = 'custom'
            self.sample = obj
        elif isinstance(obj, type(self)):
            # Copy constructor
            self.species = obj.species
            self.sample = obj.sample
        elif isinstance(obj, numbers.Number):
            # Constant number
            self.species = 'number'
            self.sample = lambda _=None: obj
        elif isinstance(obj, (np.ndarray, list, tuple)):
            # Initialize from array-like (of type number, RV, array, etc.)
            self.species = 'array'
            array = np.array([RV(item) for item in obj])

            def sample(seed):
                seed = gen_seed(seed)
                rv_samples = [rv(seed) for rv in array]
                return np.array(rv_samples)
            self.sample = sample
        else:
            # Initialize from scipy.stats random variable (or similar)
            self.species = 'scipy'
            self.sample = lambda seed=None: obj.rvs(random_state=seed)

    def __call__(self, seed=None):
        """Draw a random sample."""

        return self.sample(seed)

    def __getitem__(self, key):
        if self.species == 'composed':
            def component_fcn(*args):
                return self.arg(*args)[key]
            return RV(component_fcn, *self.args)
        else:
            return RV(self.arg[key])

    # Operators for emulating numeric types
    for p in programs_lift + programs_other + programs_unary:
        exec(p)
