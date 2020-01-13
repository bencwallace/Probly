"""
Random variables following common distributions.
"""

from ..core.random_variables import RandomVariable
from ..lib import const


# ultimately makes Distribution an instance of a specialization of the reader monad
# metaclasses needed so that subclasses also become instances
class Lift(type):
    def __call__(cls, *params, **kwargs):
        if any((isinstance(rv, RandomVariable) for rv in params)):
            return RandomDistribution(cls, *(const(rv) for rv in params))
        else:
            return super().__call__(*params, **kwargs)


class RandomDistribution(RandomVariable):
    def __init__(self, distr, *rvs):
        self.distr = distr
        self.rvs = rvs
        super().__init__()

    def _sampler(self, seed=None):
        seed = self._seed(seed)
        # need to short-circuit to sampler for testability
        return self.distr(*(rv(seed) for rv in self.rvs))._sampler(seed)


class Distribution(RandomVariable, metaclass=Lift):
    """
    A random variable given by some distribution.

    Subclassing
    -----------
    Subclasses should call `super().__init__()` in their initializers.
    A subclass should implement the method `_sampler(self, seed)`,
    which produces a random sample from a given integer seed according
    to the desired distribution.

    It is also recommended, when these quantities are relatively simple to
    compute, to override `mean(self)`, `momen(self, p)`, `cmoment(self, p)`,
    `variance(self)`, `cdf(self, x)`, and `pdf(self, x)`.

    Example
    -------
    Define a family of "shifted" uniform random variables:

    >>> import probly as pr
    >>> import numpy as np
    >>> class UnifShift(Distribution):
    ...     def __init__(self, a, b):
    ...         self.a = a + 1
    ...         self.b = b + 1
    ...         super().__init__()
    ...     def _sampler(self, seed):
    ...         np.random.seed(seed)
    ...         return np.random.uniform(self.a, self.b)
    """
