"""
Random variables following common distributions.
"""

from ..core.random_variables import RandomVariable, RandomVariableWithIndependence
from ..lib import const


# ultimately makes Distribution an instance of a specialization of the reader monad
# metaclasses needed so that subclasses also become instances
class Lift(type):
    def __call__(cls, *rvs, **kwargs):
        if any((isinstance(rv, RandomVariable) for rv in rvs)):
            return RandomDistribution(cls, *(const(rv) for rv in rvs))
        else:
            return super().__call__(*rvs, **kwargs)


class RandomDistribution(RandomVariable):
    def __init__(self, distr, *rvs):
        self.distr = distr
        self.rvs = rvs
        super().__init__(self._sampler)

    def _sampler(self, seed=None):
        seed = self._get_seed(seed)
        # need to short-circuit to sampler for testability
        return self.distr(*(rv(seed) for rv in self.rvs))._sampler(seed)


class Distribution(RandomVariableWithIndependence, metaclass=Lift):
    def __init__(self):
        op = self._sampler
        super().__init__(op)
