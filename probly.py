"""probly.py: A python module for working with random variables."""

import numbers
import numpy as np
import operator as op

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
        '   X = sampler(x)\n'
        '   return X.__{:s}__(self)'
    ).format(fcn, fcn) for fcn in num_ops_right]

programs_unary = [
    (
        'def __{:s}__(self):\n'
        '   return Lift(op.{:s})(self)'
    ).format(fcn, fcn) for fcn in num_ops_unary]


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(*argv):
        return sampler(f, *argv)

    return F


class sampler(object):
    """
    A generic random sampler.

    Basically a very general kind of random variable. Only capable of producing
    random quantities, whose explicit distribution is not necessarily known.

    Attributes:
        species (str)
        sample (function)
    """

    def __init__(self, arg, *argv):
        """Type conversion."""

        self.argv = [sampler(rv) for rv in argv]

        if isinstance(arg, type(self)):
            self.species = arg.species
            self.sample = arg.sample
            self.argv = arg.argv
        elif callable(arg):
            self.species = 'composed'

            def sample(seed=None):
                rv_samples = [rv(seed) for rv in self.argv]
                return arg(*rv_samples)
            self.sample = sample
        elif isinstance(arg, (numbers.Number, np.ndarray, list)):
            self.species = 'const'
            if isinstance(arg, list):
                arg = np.array(arg)
            self.sample = lambda _=None: arg
        else:   # Assume scipy.stats random variable
            self.species = 'scipy'
            self.sample = lambda seed=None: arg.rvs(random_state=seed)

    def __call__(self, seed=None):
        """Draw a random sample."""

        return self.sample(seed)

    # Operators for emulating numeric types
    for p in programs_lift + programs_other + programs_unary:
        exec(p)
