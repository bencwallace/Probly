import numpy as np
import math
from os import urandom
from functools import wraps


# Numpy max seed
_max_seed = 2 ** 32 - 1


def Lift(cls, f):
        """
        Lifts a function to `cls`-valued map.

        Args:
            cls (type): Specifies desired output type of lifted map.
        """

        @wraps(f)
        def F(*args):
            """
            The lifted function

            Args:
                `rvar`s and constants
            """

            F_of_args = cls.__new__(cls)

            cls.graph.add_node(F_of_args, method=f)
            edges = [(cls._cast(var), F_of_args, {'index': i})
                     for i, var in enumerate(args)]
            cls.graph.add_edges_from(edges)

            return F_of_args

        return F


def get_seed(seed=None):
    """
    Generate a random seed. If a seed is provided, returns it unchanged.

    Based on the Python implementation. A consistent approach to generating
    re-usable random seeds is needed in order to implement dependency.
    """

    if seed is not None:
        return seed

    try:
        max_bytes = math.ceil(np.log2(_max_seed) / 8)
        seed = int.from_bytes(urandom(max_bytes), 'big')
    except NotImplementedError:
        raise NotImplementedError('Seed from time not implemented.')

    return seed
