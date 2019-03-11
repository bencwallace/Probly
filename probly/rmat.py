from .core import RandomVar
from .distr import Normal
from .helpers import array


class Wigner(RandomVar):
    """
    A Wigner random matrix.

    A random symmetric matrix whose upper-diagonal entries are independent,
    identically distributed random variables.

    Parameters
    ----------
    dim : int
        The matrix dimension.
    rv : RandomVar, optional
        A random variable whose distribution the entries will share. Default is
        a standard normal random variable.
    """

    def __new__(self, dim, rv=None):
        if rv is None:
            rv = Normal(0, 1)
        # Upper-diagonal part
        arr = [[rv.copy() if i <= j else 0 for j in range(dim)]
               for i in range(dim)]

        # Lower-diagonal part
        arr = [[arr[i][j] if i <= j else arr[j][i] for j in range(dim)]
               for i in range(dim)]

        return array(arr)


class Wishart(RandomVar):
    pass
