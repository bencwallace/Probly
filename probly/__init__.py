"""
Probly implements random variables and methods to aid in their construction.

The main objects of Probly are random variable classes. These inherit from the
core class ``RandomVar``, which acts as a subclassing interface for a node in a
computational graph. Probly also provides methods to help in the formation of
new random variables.
"""


# Constructors and helpers
from .construct import Lift, array, sum
from .helpers import hist, mean

# Discrete random variables
from .distr import DUnif
from .distr import Multinomial, Bin, Ber
from .distr import NegBin, Geom
from .distr import HyperGeom, Pois

# Continuous random variables
from .distr import Gamma, ChiSquared, Exp
from .distr import Unif
from .distr import Normal, LogNormal
from .distr import Beta, PowerLaw
from .distr import F, Student_t
from .distr import Laplace, Logistic, VonMises

# Random matrices
from .rmat import Wigner, Wishart

__all__ = []

# Constructors and helpers
__all__ += ['Lift', 'array', 'sum']
__all__ += ['hist', 'mean']

# Discrete random variables
__all__ += ['DUnif']
__all__ += ['Multinomial', 'Bin', 'Ber']
__all__ += ['NegBin', 'Geom']
__all__ += ['HyperGeom', 'Pois']

# Continuous random variables
__all__ += ['Gamma', 'ChiSquared', 'Exp']
__all__ += ['Unif']
__all__ += ['Normal', 'LogNormal']
__all__ += ['Beta', 'PowerLaw']
__all__ += ['F', 'Student_t']
__all__ += ['Laplace', 'Logistic', 'VonMises']

# Random matrices
__all__ += ['Wigner', 'Wishart']
