"""
Random variables for common distributions.

Subclassing instructions:

    To define a random variable class with a desired distribution, create a
    subclass of `Distr` with a single method `sampler` whose arguments are
    `self` and the desired parameters and which samples from the distribution
    of interest. This sampler has access to the attribute `self.seed` and may
    seed the random number generator with `np.random.seed(self.seed)`.
"""

import numpy as np
from .core import rv


class Distr(rv):
    def __new__(cls, *args):
        return super().__new__(cls)

    def __init__(self, *args):
        self.params = args

        super().__init__()

    def sampler_fixed(self, seed=None):
        # Assume self.sampler does not accept seed argument
        self.seed = seed
        return self.sampler(*self.params)

    def sampler(self, *args):
        # Overload in subclass
        pass


class Unif(Distr):
    def sampler(self, a, b):
        np.random.seed(self.seed)
        return np.random.uniform(a, b)


class Ber(Distr):
    def sampler(self, p):
        np.random.seed(self.seed)
        return np.random.choice(2, p=[1 - p, p])
