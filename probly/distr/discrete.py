import numpy as np

from .distributions import Distribution


# -------------------- Discrete uniform random variable -------------------- #

class RandInt(Distribution):
    """
    A discrete uniform random variable.

    Parameters
    ----------
    a : int
        Lowest possible value.
    b : int
        Highest possible value. Default is `a + 1`.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()

    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.randint(self.a, self.b + 1)

    def mean(self, **kwargs):
        return (self.a + self.b) / 2

    def __str__(self):
        return 'RandInt({}, {})'.format(self.a, self.b)


# --------------------------- Multinomial family --------------------------- #

class Multinomial(Distribution):
    """
    A multinomial random variable.

    Parameters
    ----------
    n : int
        Number of trials.
    pvals : list or tuple, optional
        Success probabilities. Default is equal probabilities for each outcome.
    """

    def __init__(self, n, pvals=None):
        self.n = n
        if not pvals:
            self.pvals = [1 / n] * n
        else:
            self.pvals = pvals
        super().__init__()

    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.multinomial(self.n, self.pvals)

    def mean(self, **kwargs):
        return np.array([self.n * pval for pval in self.pvals])

    def __str__(self):
        return 'Multinomial({}, {})'.format(self.n, self.pvals)


class Bin(Multinomial):
    """
    A binomial random variable.

    Represents the number of successful trials out of `n` independent Bernoulli
    trials with probability of success `p`.

    Parameters
    ----------
    n : int
        Number of trials.

    p : float, optional
        probability of success.
    """

    def __init__(self, n, p=0.5):
        self.p = p
        super().__init__(n, [1 - p, p])

    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.binomial(self.n, self.p)

    def mean(self, **kwargs):
        return self.n * self.p

    def __str__(self):
        return 'Bin({}, {})'.format(self.n, self.p)


class Ber(Bin):
    """
    A Bernoulli random variable.

    Takes the value `1` with probability `p` and `0` with probability `1 - p`.

    Parameters
    ----------
    p : float, optional
        Probability that the outcome is `1`.
    """

    # Uses np.random.binomial with n = 1 (much faster than np.random.choice)
    def __init__(self, p=0.5):
        super().__init__(1, p)

    def mean(self, **kwargs):
        return self.p

    def __str__(self):
        return 'Ber({})'.format(self.p)


# ------------------------ Negative binomial family ------------------------ #

class NegBin(Distribution):
    """
    A negative binomial random variable.

    Represents the number of successes in a sequence of independent Bernoulli
    trials with probability of success `p` before `n` failures occur.

    Parameters
    ----------
    n : int
        Number of failures.

    p : float, optional
        Probability of success.
    """

    def __init__(self, n, p=0.5):
        self.n = n
        self.p = p
        super().__init__()

    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.negative_binomial(self.n, self.p)

    def mean(self, **kwargs):
        return self.n * (1 - self.p) / self.p

    def __str__(self):
        return 'NegBin({}, {})'.format(self.n, self.p)


class Geom(NegBin):
    """
    A geometric random variable.

    Represents the number of independent Bernoulli trials needed before a trial
    is successful.

    Parameters
    ----------
    p : float, optional
        Probability of success.
    """

    def __init__(self, p=0.5):
        super().__init__(1, p)

    # Faster than using np.random.negative_binomial
    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.geometric(self.p)

    def mean(self, **kwargs):
        return 1 / self.p

    def __str__(self):
        return 'Geom({})'.format(self.p)


# --------------------- Other discrete random variables --------------------- #

class HyperGeom(Distribution):
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
        super().__init__()

    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.hypergeometric(self.ngood, self.nbad, self.nsample)

    def mean(self, **kwargs):
        return self.ngood * self.nbad / self.nsample

    def __str__(self):
        return 'HyperGeom({}, {},'\
               ' {})'.format(self.ngood, self.nbad, self.nsample)


class Pois(Distribution):
    """
    A Poisson random variable.

    Parameters
    ----------
    rate : float, optional
        The rate parameter.
    """

    def __init__(self, rate=1):
        self.rate = rate
        super().__init__()

    def _sampler(self, seed):
        np.random.seed(seed)
        return np.random.poisson(self.rate)

    def mean(self, **kwargs):
        return self.rate

    def __str__(self):
        return 'Pois({})'.format(self.rate)
