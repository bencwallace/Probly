"""
Families of random variables defined from a distribution.
"""

import numpy as np
from .core import RandomVar

# ======================== Discrete random variables ======================== #


# -------------------- Discrete uniform random variable -------------------- #

class DUnif(RandomVar):
    """
    A discrete uniform random variable.

    Parameters
    ----------
    a : int
        Lowest possible value.
    b : int
        Highest possible value.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.randint(self.a, self.b + 1)


# --------------------------- Multinomial family --------------------------- #

class Multinomial(RandomVar):
    """
    A multinomial random variable.

    Parameters
    ----------
    n : int
        Number of trials
    pvals : list or tuple
        Success probabilities
    """

    def __init__(self, n, pvals):
        self.n = n
        self.pvals = pvals

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.multinomial(self.n, self.pvals)


class Bin(Multinomial):
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
        super().__init__(self, n, [1 - p, p])

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


# ------------------------ Negative binomial family ------------------------ #

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


# --------------------- Other discrete random variables --------------------- #

class HyperGeom(RandomVar):
    """
    A hypergeometric random variable.

    Parameters
    ----------
    ngood : int
    nbad : int
    nsample : int
    """

    def __init__(self, ngood, nbad, nsample):
        self.ngood = ngood
        self.nbad = nbad
        self.nsample = nsample

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.hypergeometric(self.ngood, self.nbad, self.nsample)


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


# ----------------------- Continuous random variables ----------------------- #


# ------------------------------ Gamma family ------------------------------ #

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


# ------------------------ Uniform random variables ------------------------ #

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


# ------------------- Stable random variables ------------------- #

class Normal(RandomVar):
    """
    A normal random variable.

    Parameters
    ----------
    mean : float
        Mean.
    sd : float
        Standard deviation.
    """

    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.normal(self.mean, self.sd)


class LogNormal(RandomVar):
    """
    A log-normal random variable.

    Parameters
    ----------
    mean : float
    sd : float
    """

    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.lognormal(self.mean, self.sd)


# --------------------- Beta distribution and power law --------------------- #

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


class PowerLaw(RandomVar):
    """
    A random variable following a power law.

    Parameters
    ----------
    power : float
    """

    def __init__(self, power):
        self.power = power

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.power(self.power)


# ------------------------ F and t random variables ------------------------ #

class F(RandomVar):
    """
    An F random variable.

    Parameters
    ----------
    d1 : int
    d2 : int
    """

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.f(self.d1, self.d2)


class Student_t(RandomVar):
    """
    A Student's t random variable.

    Parameters
    ----------
    deg : float
    """

    def __init__(self, deg):
        self.deg = deg

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.standard_t(self.deg)


# -------------------- Other continuous random variables -------------------- #

class Laplace(RandomVar):
    """
    A Laplace random variable.

    Parameters
    ----------
    loc : float
    scale : float
    """

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.laplace(self.loc, self.scale)


class Logistic(RandomVar):
    """
    A logistic random variable.

    Parameters
    ----------
    loc : float
    scale : float
    """

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.logistic(self.loc, self.scale)


class Pareto(RandomVar):
    """
    A Pareto random variable.

    Parameters
    ----------
    shape : float
    """

    def __init__(self, shape):
        self.shape = shape

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.pareto(self.shape)


class VonMises(RandomVar):
    """
    A von Mises random variable.

    Parameters
    ----------
    mean : float
    kappa : float
    """

    def __init__(self, mean, kappa):
        self.mean = mean
        self.kappa = kappa

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.vonmises(self.mean, self.kappa)


class Weibull(RandomVar):
    """
    A Weibull random variable.

    Parameters
    ----------
    shape : float
    """

    def __init__(self, shape):
        self.shape = shape

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.weibull(self.shape)
