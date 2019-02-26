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


class Bin(Distr):
    def sampler(self, n, p, seed=None):
        np.random.seed(seed)
        return np.random.binomial(n, p)


class Ber(Bin):
    def __new__(cls, p):
        return super().__new__(cls, 1, p)

    # Overload sampler for efficiency (to do: test if there's a difference)
    def sampler(self, p, seed=None):
        np.random.seed(seed)
        return np.random.choice([0, 1], p=[1 - p, p])


# # Alternate definition
# class Ber(Bin):
#     def __new__(cls, p):
#         X = super().__new__(cls, 1, p)
#         return X

#     def __init__(self, p):
#         super().__init__(1, p)


class Beta(Distr):
    def sampler(self, alpha, beta, seed=None):
        np.random.seed(seed)
        return np.random.beta(alpha, beta)


class Gamma(Distr):
    def sampler(self, shape, rate=None, scale=None, seed=None):
        assert rate is not None or scale is not None,\
            'Specify rate or scale'

        if scale is None and rate is not None:
            scale = 1 / float(rate)

        np.random.seed(seed)
        return np.random.gamma(shape, scale)


class ChiSquared(Gamma):
    def __new__(cls, k):
        shape = float(k) / 2
        scale = 2
        rate = 1 / scale

        return super().__new__(cls, shape, rate, scale)

    def sampler(self, k, seed=None):
        np.random.seed(seed)
        return np.random.chisquare(k)


class Exp(Gamma):
    def __new__(cls, rate):
        shape = 1
        scale = 1 / float(rate)

        return super().__new__(cls, shape, rate, scale)

    def sampler(self, rate, seed=None):
        np.random.seed(seed)
        return np.random.exponential(rate)


class NegBin(Distr):
    def sampler(self, n, p, seed=None):
        np.random.seed(seed)
        return np.random.negative_binomial(n, p)


class Geom(NegBin):
    def __new__(cls, p):
        return super().__new__(cls, 1, p)

    def sampler(self, p, seed=None):
        np.random.seed(seed)
        return np.random.geometric(p)


class Pois(Distr):
    def sampler(self, rate, seed=None):
        np.random.seed(seed)
        return np.random.poisson(rate)


class Normal(Distr):
    def sampler(self, mean, std, seed=None):
        np.random.seed(seed)
        return np.random.normal(mean, std)
