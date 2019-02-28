"""
Random variables for common distributions.
"""

import numpy as np
from .core import rv


class Distr(rv):
    """
    A random variable specified by its distribution.

    The `Distr` class is intended to be subclassed for defining new random
    variables.

    Notes
    -----
    To define a new random variable class with a desired distribution, create a
    subclass of `Distr` with at least a single method `sampler` that accepts
    the single argument `seed` with default value `None`.
    The `sampler` method should output values from the desired distribution.

    It is advised to properly make use of `seed` in the definition of
    `sampler`.

    Example
    -------
    The following defines a uniform random variable on the interval
    `[a + 1, b + 2]`:

    >>> import numpy as np
    >>> import probly as pr
    >>> class UnifPlus(Distr):
    ...     def __init__(self, a, b):
    ...         self.a = a + 1
    ...         self.b = b + 1
    ...     def sampler(self, seed=None):
    ...         np.random.seed(seed)
    ...         return np.random.uniform(self.a, self.b)
    """

    # Protection from the perils of sub-classing rv directly
    def __new__(cls, *args):
        # Create bare rv (add to graph)
        obj = super().__new__(cls)

        return obj


class Unif(Distr):
    """
    A uniform random variable.

    Parameters
    ----------
        a : float
            Left endpoint of the support interval.
        b : float
            Right endpoint of the support interval.

    Attributes
    ----------
        a : float
            Left endpoint of the support interval.
        b : float
            Right endpoint of the support interval.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.uniform(self.a, self.b)


class Bin(Distr):
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.binomial(self.n, self.p)


class Ber(Bin):
    # Uses np.random.binomial with n = 1 (much faster than np.random.choice)
    def __init__(self, p):
        super().__init__(1, p)


class Beta(Distr):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.beta(self.alpha, self.beta)


class Gamma(Distr):
    def __init__(self, shape, rate=None, scale=None):
        self.shape = shape
        self.rate = rate
        self.scale = scale

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.gamma(self.shape, self.scale)


class ChiSquared(Gamma):
    def __init__(self, k):
        self.k = k

        shape = float(k) / 2
        scale = 2
        rate = 1 / scale

        super().__init__(shape, rate, scale)

    # Much faster than using np.random.gamma
    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.chisquare(self.k)


class Exp(Gamma):
    def __init__(self, rate):
        shape = 1
        scale = 1 / float(rate)

        super().__init__(shape, rate, scale)

    # A bit faster than using np.random.gamma
    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.exponential(self.rate)


class NegBin(Distr):
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.negative_binomial(self.n, self.p)


class Geom(NegBin):
    def __init__(self, p):
        super().__init__(1, p)

    # Faster than using np.random.negative_binomial
    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.geometric(self.p)


class Pois(Distr):
    def __init__(self, rate):
        self.rate = rate

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.poisson(self.rate)


class Normal(Distr):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.normal(self.mean, self.std)
