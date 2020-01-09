def mean(rv, max_iter=int(1e5), tol=1e-5):
    return rv.mean(max_iter=max_iter, tol=tol)


def moment(rv, p, **kwargs):
    return rv.moment(p, **kwargs)


def cmoment(rv, p, **kwargs):
    return rv.cmoment(p, **kwargs)


def variance(rv, **kwargs):
    return rv.variance(**kwargs)


def cdf(rv, x, **kwargs):
    return rv.cdf(x, **kwargs)


def pdf(rv, x, dx=1e-5, **kwargs):
    return rv.pdf(x, dx, **kwargs)
