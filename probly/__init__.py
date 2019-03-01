"""
ProbLy

A python package for the symbolic computation of random variables.
"""

from .randomvar import array, Lift

from .distr import Distr
from .distr import Bin, Ber
from .distr import Gamma, ChiSquared, Exp
from .distr import NegBin, Geom
from .distr import Beta, Unif, Pois, Normal

__all__ = ['array', 'Lift', 'Distr', 'Bin', 'Ber', 'Gamma', 'ChiSquared',
           'Exp', 'NegBin', 'Geom', 'Beta', 'Unif', 'Pois', 'Normal']
