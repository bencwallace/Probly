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
    subclass of `Distr` with at least a single method `_sampler` that accepts
    the single argument `seed` with default value `None`.
    The `_sampler` method should output values from the desired distribution.

    It is advised to properly make use of `seed` in the definition of
    `_sampler`.

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
    ...     def _sampler(self, seed=None):
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

    Note
    ----
    Attributes are the same as parameters.
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

    Attributes
    ----------
    p : float
        Probability that the outcome is `1`.
    n : int
        Parameter `n` inherited from the corresponding binomial distribution.
        Equal to `1`.
    """

    # Uses np.random.binomial with n = 1 (much faster than np.random.choice)
    def __init__(self, p):
        super().__init__(1, p)


class Beta(Distr):
    """
    A beta random variable.

    Parameters
    ----------
    alpha : (float)
        First shape parameter.

    beta : (float)
        Second shape parameter.

    Note
    ----
    Attributes are the same as parameters.
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

    Attributes
    ----------
    shape : float
        Shape parameter.
    rate : float
        Rate parameter.
    scale : float
        Scale parameter.
    """

    def __init__(self, shape, rate=None, scale=None):
        self.shape = shape
        self.rate = rate
        self.scale = scale

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

    Attributes
    ----------
    k : float
        Number of degrees of freedom.
    shape : float
        Shape parameter inherited from the corresponding gamma distribution.
        Equal to `k / 2`.
    rate : float
        Rate parameter inherited from the corresponding gamma distribution.
        Equal to `2`.
    scale : float
        Scale parameter inherited from the corresponding gamma distribution.
        Equal to `1 / 2`.
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

    Attributes
    ----------
    rate : float
        Rate parameter.
    shape : int
        Shape parameter inherited from the corresponding gamma distribution.
        Equal to `1`.
    scale : float
        Scale parameter inherited from the corresponding gamma distribution.
        Equal to `1 / rate`.
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

    Note
    ----
    Attributes are the same as parameters.
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

    Attributes
    ----------
    p : float
        Probability of success.
    n : Parameter `n` inherited from the corresponding negative binomial
        distribution. Equal to `1`.
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

    Note
    ----
    Attributes are the same as parameters.
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

    Note
    ----
    Attributes are the same as parameters.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.normal(self.mean, self.std)
