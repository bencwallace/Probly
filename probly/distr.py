"""
Random variables for common distributions.

Subclassing instructions:

    To define a random variable class with a desired distribution, create a
    subclass of `Distr` with a single method `sampler(self, seed)`, which
    samples from the desired distribution. Seeding according to `seed` is taken
    care of by the `rv` class. Initialization of the subclass should take a
    list of parameters, which can be recovered in the `sampler` method from
    `self.params`.
"""

import numpy as np
from . import rv


class Distr(rv):
    def __init__(self, *args):
        self.params = args
        # self.params = rv._cast(args)

        def sampler(*args):
            return self.sampler()
        # super().__init__(sampler, self.params)
        super().__init__(sampler)

    def sampler(self, seed=None):
        # Overload in subclass
        pass


# To do: allow a, b random
class Unif(Distr):
    def sampler(self):
        a, b = self.params
        return np.random.uniform(a, b)


class Ber(Distr):
    def sampler(self, seed=None):
        p, = self.params
        return np.random.choice(2, p=[1 - p, p])
