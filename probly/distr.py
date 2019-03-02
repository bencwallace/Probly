"""
Random variables for common distributions.
"""

import numpy as np

from .core import RNG
from .randomvar import RandomVar


class Distr(RandomVar):
    """
    A random variable or family therefore specified by a distribution.

    The `Distr` class is intended to be subclassed for defining new random
    variables and families of random variables.

    Notes
    -----
    To define a new distribution, create a
    subclass of `Distr` with at least a single method `_sampler` that accepts
    the single argument `seed` with default value `None`.
    The `_sampler` method should output values from the desired distribution.
    It is important to make consistent use of `seed` in the definition of
    `_sampler`.

    In order to define a parameterized family of random variables, one should
    additionally define an `__init__` constructor that accepts the desired
    parameters as arguments and stores their values as object attributes.
    These values can then be used by the `_sampler` method.

    The `_sampler` method is marked as private and should be treated as such.
    Rather than calling `_sampler` directly, one should call the random
    variable instance itself, which is callable by inheritance. Doing so
    automatically takes care of ensuring independence of different instances
    of a random variable.

    Example
    -------
    Define a family of "shifted" uniform random variables:

    >>> import numpy as np
    >>> import probly as pr
    >>> class UnifShift(pr.Distr):
    ...     def __init__(self, a, b):
    ...         self.a = a + 1
    ...         self.b = b + 1
    ...     def _sampler(self, seed=None):
    ...         np.random.seed(seed)
    ...         return np.random.uniform(self.a, self.b)

    Instantiate a random variable from this family with support `[1, 2]` and
    sample from its distribution:

    >>> X = UnifShift(0, 1)
    >>> X()
    """

    # Protection from the perils of sub-classing RandomVar directly
    def __new__(cls, *args, **kwargs):
        # Create bare RandomVar (add to graph)
        return super().__new__(cls, 'sampler', RNG)

    def __call__(self, seed=None):
        return self._sampler((RNG(seed) + self._offset) % self._max_seed)


class Unif(Distr):
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


class Bin(Distr):
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


class Beta(Distr):
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


class Gamma(Distr):
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


class NegBin(Distr):
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


class Pois(Distr):
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


class Normal(Distr):
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
