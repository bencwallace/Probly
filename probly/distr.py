"""
Random variables for common distributions.
"""

import numpy as np
from .core import RandomVar


class Unif(RandomVar):
    """
    A uniform random variable.

    Parameters
    ----------
    a : float
        Left endpoint of the support interval.
    b : float
        Right endpoint of the selfupport inteRandomVaral.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.uniform(self.a, self.b)


class Bin(RandomVar):
    """
    A binomial random variable.

    Represents the number of successful trials out of `n` independent Bernoulli
    trials with probability of success `p`.

    Parameters
    ----------
    n : int
        Number of trials.

    p : float
        probability of success.
    """

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.binomial(self.n, self.p)


class Ber(Bin):
    """
    A Bernoulli random variable.

    Takes the value `1` with probability `p` and `0` with probability `1 - p`.

    Parameters
    ----------
    p : float
        Probability that the outcome is `1`.
    """

    # Uses np.random.binomial with n = 1 (much faster than np.random.choice)
    def __init__(self, p):
        super().__init__(1, p)


class Beta(RandomVar):
    """
    A beta random variable.

    Parameters
    ----------
    alpha : float
        First shape parameter.

    beta : float
        Second shape parameter.
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.beta(self.alpha, self.beta)


class Gamma(RandomVar):
    """
    A gamma random variable.

    Supports the two common parameterizations of the gamma distribution:
    the `(shape, rate)` parameterization and the `(shape, scale)`
    parameterization. The `shape` parameter and at least one of the `rate` or
    `scale` parameters must be specified.

    Parameters
    ----------
    shape : float
        Shape parameter.
    rate : float, optional if `scale` specified
        Rate parameter.
    scale : float, optional if `rate` specified
        Scale parameter.
    """

    def __init__(self, shape, rate=None, scale=None):
        self.shape = shape

        if scale is not None:
            self.scale = scale
            self.rate = 1 / scale
        else:
            self.rate = rate
            self.scale = 1 / rate

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.gamma(self.shape, self.scale)


class ChiSquared(Gamma):
    """
    A chi squared random variable.

    Parameters
    ----------
    k : float
        Number of degrees of freedom.
    """

    def __init__(self, k):
        self.k = k

        shape = float(k) / 2
        scale = 2
        rate = 1 / scale

        super().__init__(shape, rate, scale)

    # Much faster than using np.random.gamma
    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.chisquare(self.k)


class Exp(Gamma):
    """
    An exponential random variable.

    Parameters
    ----------
    rate : float
        Rate parameter.
    """

    def __init__(self, rate):
        shape = 1
        scale = 1 / float(rate)

        super().__init__(shape, rate, scale)

    # A bit faster than using np.random.gamma
    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.exponential(self.rate)


class NegBin(RandomVar):
    """
    A negative binomial random variable.

    Represents the number of successes in a sequence of independent Bernoulli
    trials with probability of success `p` before `n` failures occur.

    Parameters
    ----------
    n : int
        Number of failures.

    p : Probability of success.
    """

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.negative_binomial(self.n, self.p)


class Geom(NegBin):
    """
    A geometric random variable.

    Represents the number of independent Bernoulli trials needed before a trial
    is successful.

    Parameters
    ----------
    p : float
        Probability of success.
    """

    def __init__(self, p):
        super().__init__(1, p)

    # Faster than using np.random.negative_binomial
    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.geometric(self.p)


class Pois(RandomVar):
    """
    A Poisson random variable.

    Parameters
    ----------
    rate : float
        The rate parameter.
    """

    def __init__(self, rate):
        self.rate = rate

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.poisson(self.rate)


class Normal(RandomVar):
    """
    A normal random variable.

    Parameters
    ----------
    mean : float
        Mean.
    std : float
        Standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.normal(self.mean, self.std)
