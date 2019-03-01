"""
======
ProbLy
======

A python package for the symbolic computation of random variables.
"""

from .randomvar import array, Lift

# Binomial-type distributions
from .distr import Bin, Ber

# Gamma-type distributions
from .distr import Gamma, ChiSquared, Exp

# Negative-binomial-type distributions
from .distr import NegBin, Geom

# Other distributions
from .distr import Beta, Unif, Pois, Normal

__all__ = ['array', 'Lift']
__all__ += ['Bin', 'Ber']
__all__ += ['Gamma', 'ChiSquared', 'Exp']
__all__ += ['NegBin', 'Geom']
__all__ += ['Beta', 'Unif', 'Pois', 'Normal']
