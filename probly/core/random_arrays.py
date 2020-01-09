import numpy as np

from .random_variables import RandomVariable
from ..distr import Normal


def random_array(rv, shape):
    array = np.array([rv.copy() for _ in np.nditer(np.ndarray(shape))]).reshape(shape)
    return RandomArray(array)


class RandomArray(RandomVariable):
    def __init__(self, array):
        def op(*inputs):
            return np.array(inputs).reshape(np.shape(array))
        super().__init__(op, *array.flatten())

        self.shape = np.array(array).shape


class Wigner(RandomArray):
    """
    A Wigner random matrix.

    A random symmetric matrix whose upper-diagonal entries are independent,
    identically distributed random variables.

    Parameters
    ----------
    dim : int
        The matrix dimension.
    rv : RandomVariable, optional
        A random variable whose distribution the entries will share. Default is
        a standard normal random variable.
    """

    def __init__(self, dim, rv=None):
        self.dim = dim
        if rv is None:
            rv = Normal()
        self.rv = rv

        # Upper-diagonal part
        array = [[rv.copy() if i <= j else 0
                  for j in range(dim)] for i in range(dim)]

        # Lower-diagonal part
        array = [[array[i][j] if i <= j else array[j][i]
                  for j in range(dim)] for i in range(dim)]

        super().__init__(array)

    def __str__(self):
        return 'Wigner({}, {})'.format(self.dim, self.rv)


class Wishart(RandomArray):
    """
    A Wishart random matrix.

    An `n` by `n` random symmetric matrix obtained as the (matrix) product of
    an `m` by `n` random matrix with independent, identically distributed
    entries, and its transpose.

    Parameters
    ----------
    m : int
        The first dimension parameter.
    n : int
        The second dimension parameter.
    rv : RandomVariable, optional
        A random variable.


    Attributes
    ----------
    lambda_ : float
        The ratio `m / n`.
    """

    def __init__(self, m, n, rv=None):
        self.m = m
        self.n = n
        self.lambda_ = m / n
        if rv is None:
            rv = Normal()
        self.rv = rv

        rect = np.array([[rv.copy() for _ in range(n)] for _ in range(m)])
        square = np.dot(rect.T, rect)
        super().__init__(square)

    def __str__(self):
        return 'Wishart({}, {}, {})'.format(self.m, self.n, self.rv)
