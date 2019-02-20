"""
Extensions to scipy.stats

Example 1
---------
Generate a chi-squared distribution from a normal distribution:

    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scirandom import composable

    def square(x):
        return x ** 2
    square = composable(square)

    n = stats.norm()
    chisq = square(n)

    samples = [chisq.rvs() for _ in range(1000)]
    _ = plt.hist(samples, bins=20)
    plt.show()

Example 2
---------
Generate the distribution of the largest eigenvalue of a Wishart matrix.

    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    from scirandom import composable

    def largest_lambda(w):
        return np.max(np.linalg.eig(w)[0])
    largest_lambda = composable(largest_lambda)

    W = stats.wishart(scale=np.eye(2))
    eig = largest_lambda(W)

    samples = [eig.rvs() for _ in range(1000)]
    _ = plt.hist(samples, bins=20)
    plt.show()
"""


class sampler():
    """
    A sampler

    Basically a very general kind of random variable. Only capable of producing
    random quantities, whose explicit distribution is not necessarily known.
    """

    def __init__(self, f, *argv):
        self.argv = argv
        self.f = f

    def rvs(self):
        samples = [var.rvs() for var in self.argv]
        return self.f(*samples)


def composable(f):
    """Makes a function act on random variables."""

    def F(X):
        return sampler(f, X)

    return F
