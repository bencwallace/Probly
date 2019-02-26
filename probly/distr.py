"""
Random variables for common distributions.

Subclassing instructions:

    To define a random variable class with a desired distribution, create a
    subclass of `Distr` with a single method `sampler` whose arguments are
    `self`, the desired parameters, and `seed` with default value `None`.
    The `sampler` method should output values from the desired distribution.
    It is import not to forget to include `seed=None` in the list of arguments
    and, more importantly, to make use of this seed in the definition of
    `sampler`.

    The new random variable is then initialized with arguments given by the
    desired parameters (in the order determined by `sampler`).
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
        return self.sampler(*self.params, seed=seed)

    def sampler(self, *args):
        # Overload in subclass
        pass


class Unif(Distr):
    def sampler(self, a, b, seed=None):
        np.random.seed(seed)
        return np.random.uniform(a, b)


class Ber(Distr):
    def sampler(self, p, seed=None):
        np.random.seed(seed)
        return np.random.choice(2, p=[1 - p, p])
