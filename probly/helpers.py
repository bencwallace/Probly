import numpy as np
import math
from os import urandom
from functools import wraps


# Numpy max seed
_max_seed = 2 ** 32 - 1


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
