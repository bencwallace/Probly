"""
Random variables following common distributions.
"""

from ..core.random_variables import RandomVariable, RandomVariableWithIndependence


class Lift(type):
    def __call__(cls, *rvs, **kwargs):
        if any((isinstance(rv, RandomVariable) for rv in rvs)):
            return RandomDistribution(cls, *rvs)
        else:
            return super().__call__(*rvs, **kwargs)


class Distribution(RandomVariableWithIndependence, metaclass=Lift):
    def __init__(self):
        op = self._sampler
        super().__init__(op)


class RandomDistribution(RandomVariable):
    def __init__(self, distr, *rvs):
        self.distr = distr
        self.rvs = rvs
        super().__init__(self._sampler)

    def _sampler(self, seed=None):
        seed = self._get_seed(seed)
        return self.distr(*(rv(seed) for rv in self.rvs))(seed)
