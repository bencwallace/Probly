def Lift(f):
    """Lifts a function to the composition map between random variables."""

    @wraps(f)
    def F(*args):
        """
        The lifted function

        Args:
            `rvar`s and constants
        """

        return rvar._compose(f, *args)

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
        raise

    return seed