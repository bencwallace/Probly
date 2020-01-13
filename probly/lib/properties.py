def mean(rv, max_iter=int(1e5), tol=1e-5):
    """
    Returns the mean of `rv`.

    In general computed numerically using up to `max_iter`
    successive approximations or until these approximations no longer
    change by more than `tol`. However, subclasses of `RandomVariable`
    may override the `mean` to produce an exact value.

    :param rv: RandomVariable
    :param max_iter: int
    :param tol: float
    """
    return rv.mean(max_iter=max_iter, tol=tol)


def variance(rv, *args, **kwargs):
    """
    Returns the variance `rv`.

    In general computed using `mean` but may be overridden.

    :param rv: RandomVariable
    """
    return rv.variance(*args, **kwargs)


def cdf(rv, x, *args, **kwargs):
    """
    Returns the value of the cumulative distribution function of `rv` evaluated at `x`.

    In general computed using `RandomVariable.mean` but may be overridden.

    :param rv: RandomVariable
    :param x: float
    :return: float
    """
    return rv.cdf(x, *args, **kwargs)
