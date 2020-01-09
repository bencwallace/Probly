"""
Probly implements random variables and methods to aid in their construction.

The main objects of Probly are random variable classes. These inherit from the
core class ``RandomVariable``, which acts as a subclassing interface for a node
in a computational graph. Probly also provides methods to help in the formation
of new random variables.
"""


from .core import *
from .distr import *
from .lib import *

__all__ = []
__all__ += ['random_array', 'RandomArray', 'Wigner', 'Wishart']
__all__ += ['const', 'hist', 'lift']
__all__ += ['mean', 'moment', 'cmoment', 'variance', 'cdf', 'pdf']

# Discrete random variables
__all__ = ['RandInt']
__all__ += ['Multinomial', 'Bin', 'Ber']
__all__ += ['NegBin', 'Geom']
__all__ += ['HyperGeom', 'Pois']

# Continuous random variables
__all__ += ['Gamma', 'ChiSquared', 'Exp']
__all__ += ['Unif']
__all__ += ['Normal', 'LogNormal']
__all__ += ['Beta', 'PowerLaw']
__all__ += ['F', 'StudentT']
__all__ += ['Laplace', 'Logistic', 'VonMises']
