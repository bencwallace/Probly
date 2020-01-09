import numpy as np

from .distributions import Distribution


# ------------------------------ Gamma family ------------------------------ #

class Gamma(Distribution):
    """
    A gamma random variable.

    Supports the two common parameterizations of the gamma distribution:
    the `(shape, rate)` parameterization and the `(shape, scale)`
    parameterization. The `shape` parameter and at least one of the `rate` or
    `scale` parameters must be specified.

    Parameters
    ----------
    shape : float, optional
        Shape parameter.
    scale : float, optional if `rate` specified
        Scale parameter.
    """

    def __init__(self, shape=1, scale=None):
        self.shape = shape
        self.scale = scale
        self.rate = 1 / scale
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.gamma(self.shape, self.scale)

    def mean(self, **kwargs):
        return self.shape * self.scale

    def __str__(self):
        return 'Gamma(shape={}, scale={})'.format(self.shape, self.scale)


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

        super().__init__(shape, scale)

    # Much faster than using np.random.gamma
    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.chisquare(self.k)

    def mean(self, **kwargs):
        return self.k

    def __str__(self):
        return 'ChiSquared({})'.format(self.k)


class Exp(Gamma):
    """
    An exponential random variable.

    Parameters
    ----------
    rate : float, optional
        Rate parameter.
    """

    def __init__(self, rate=1):
        shape = 1
        scale = 1 / float(rate)

        super().__init__(shape, scale)

    # A bit faster than using np.random.gamma
    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.exponential(self.rate)

    def mean(self, **kwargs):
        return 1 / self.rate

    def __str__(self):
        return 'Exp({})'.format(self.rate)


# ------------------------ Uniform random variables ------------------------ #

class Unif(Distribution):
    """
    A uniform random variable.

    Parameters
    ----------
    a : float, optional
        Left endpoint of the support interval.
    b : float, optional
        Right endpoint of the selfupport inteDistributional.
    """

    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.uniform(self.a, self.b)

    def mean(self, **kwargs):
        return (self.a + self.b) / 2

    def __str__(self):
        return 'Unif({}, {})'.format(self.a, self.b)


# ------------------- Normal and related random variables ------------------- #

class Normal(Distribution):
    """
    A normal random variable.

    Parameters
    ----------
    mean : float, optional
        Mean.
    cov : float, optional
        Covariance matrix (variance if 1-dimensional).
    dim : int, optional
        Dimension of the ambient space.
    """

    def __init__(self, mean=0, cov=1, dim=1):
        self.dim = dim
        self._mean = mean
        self.cov = cov
        self.shape = (dim, dim)

        if dim > 1:
            if mean == 0:
                self._mean = np.array([0] * dim)
            if cov == 1:
                self.cov = np.eye(dim)

        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        if self.dim == 1:
            return np.random.normal(self._mean, np.sqrt(self.cov))
        else:
            return np.random.multivariate_normal(self._mean,
                                                 self.cov, self.dim)

    def mean(self, **kwargs):
        return self._mean

    def __str__(self):
        return 'Normal({}, {}, {})'.format(self.mean, self.cov, self.dim)


class LogNormal(Distribution):
    """
    A log-normal random variable.

    Parameters
    ----------
    mean : float, optional
    sd : float, optional
    """

    def __init__(self, mean=0, sd=1):
        self.mean = mean
        self.sd = sd
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.lognormal(self.mean, self.sd)

    def mean(self, **kwargs):
        return np.exp(self.mean + self.sd ** 2 / 2)

    def __str__(self):
        return 'LogNormal({}, {})'.format(self.mean, self.sd)


# --------------------- Beta distribution and power law --------------------- #

class Beta(Distribution):
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
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.beta(self.alpha, self.beta)

    def mean(self, **kwargs):
        return self.alpha / (self.alpha + self.beta)

    def __str__(self):
        return 'Beta({}, {})'.format(self.alpha, self.beta)


class PowerLaw(Distribution):
    """
    A random variable following a power law.

    Parameters
    ----------
    power : float
        The power determining the rate of decay.
    """

    def __init__(self, power):
        self.power = power
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.power(self.power)

    def mean(self, **kwargs):
        return self.power / (self.power + 1)

    def __str__(self):
        return 'PowerLaw({})'.format(self.power)


# ------------------------ F and t random variables ------------------------ #

class F(Distribution):
    """
    An F random variable.

    Parameters
    ----------
    d1 : int
        The first degree of freedom parameter.
    d2 : int
        The second degree of freedom parameter.
    """

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.f(self.d1, self.d2)

    def mean(self, **kwargs):
        if self.d2 <= 2:
            return float('inf')

        return self.d2 / (self.d2 - 2)

    def __str__(self):
        return 'F({}, {})'.format(self.d1, self.d2)


class StudentT(Distribution):
    """
    A Student's t random variable.

    Parameters
    ----------
    deg : float
        The degree.
    """

    def __init__(self, deg):
        self.deg = deg
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.standard_t(self.deg)

    def mean(self, **kwargs):
        if self.deg <= 1:
            return float('inf')

        return 0

    def __str__(self):
        return 't({})'.format(self.deg)


# -------------------- Other continuous random variables -------------------- #

class Laplace(Distribution):
    """
    A Laplace random variable.

    Parameters
    ----------
    loc : float, optional
        The location parameter.
    scale : float, optional
        The scale parameter.
    """

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.laplace(self.loc, self.scale)

    def mean(self, **kwargs):
        return self.loc

    def __str__(self):
        return 'Laplace({}, {})'.format(self.loc, self.scale)


class Logistic(Distribution):
    """
    A logistic random variable.

    Parameters
    ----------
    loc : float, optional
        The location parameter.
    scale : float, optional
        The scale parameter.
    """

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.logistic(self.loc, self.scale)

    def mean(self, **kwargs):
        return self.loc

    def __str__(self):
        return 'Logistic({}, {})'.format(self.loc, self.scale)


class VonMises(Distribution):
    """
    A von Mises random variable.

    Parameters
    ----------
    mean : float, optional
        The mean.
    kappa : float, optional
        The kapp parameter.
    """

    def __init__(self, mean=0, kappa=1):
        self.mean = mean
        self.kappa = kappa
        super().__init__()

    def _sampler(self, seed=None):
        np.random.seed(seed)
        return np.random.vonmises(self.mean, self.kappa)

    def mean(self, **kwargs):
        return self.mean

    def __str__(self):
        return 'VonMises({}, {})'.format(self.mean, self.kappa)
