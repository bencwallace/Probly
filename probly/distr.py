"""
Random variables for common distributions.

Subclassing instructions:

    To define a random variable class with a desired distribution, create a
    subclass of `Distr` with a single method `sampler` whose are `self` and the
    desired parameters and which samples from the distribution of interest.
    The sampler must be based on the `numpy` random number generator
"""

import numpy as np
from .core import rv


class Distr(rv):
    def __init__(self, *args):
        self.params = args
        # self.params = rv._cast(args)

        def sampler(*args):
            return self.sampler()
        # super().__init__(sampler, self.params)
        super().__init__(sampler)

    def sampler_fixed(self):
        return self.sampler(*self.params)

    def sampler(self, *args):
        # Overload in subclass
        pass


class Unif(Distr):
    def sampler(self, a, b):
        # a, b = self.params
        return np.random.uniform(a, b)


class Ber(Distr):
    def sampler(self, p):
        # p, = self.params
        return np.random.choice(2, p=[1 - p, p])
