"""probly.py: A python module for probability."""

from copy import copy, deepcopy
import numbers
import numpy as np


class sampler():
    """
    A sampler.

    Basically a very general kind of random variable. Only capable of producing
    random quantities, whose explicit distribution is not necessarily known.

    Attributes:
        species (str)
        sample (function)
    """

    def __init__(self, arg, *argv):
        """
        Arguments:
            f : Either a function, scipy.stats RV, constant, or sampler.
        """

        self.argv = [sampler(rv) for rv in argv]

        if isinstance(arg, type(self)):
            print('Init from sampler')
            self.species = arg.species
            self.sample = arg.sample
            self.argv = arg.argv
        elif callable(arg):
            print('Init from callable')
            self.species = 'composed'
            self.fcn = deepcopy(arg)

            def sample(seed=None):
                rv_samples = [rv(seed) for rv in self.argv]
                return self.fcn(*rv_samples)
            self.sample = sample
        elif isinstance(arg, (numbers.Number, np.ndarray)):
            self.species = 'const'
            self.sample = lambda _=None: arg
        else:   # Assume scipy.stats random variable
            self.species = 'scipy'
            self.sample = lambda seed=None: arg.rvs(random_state=seed)

    def __call__(self, seed=None):
        return self.sample(seed)

    def __add__(self, x):
        return Add(self, x)


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(X):
        return sampler(f, X)

    return F


def add(x, y):
    return x + y


Add = Lift(add)
