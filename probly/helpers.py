"""
Methods for analyzing random variables.
"""


import matplotlib.pyplot as plt


def hist(rv, num_samples, bins=20, density=True):
    """
    Plots a histogram from samples of a random variable.

    Parameters
    ----------
    rv : RandomVar
        A random variable.
    num_samples : int
        The number of samples to draw from `rv`.
    bins : int
        The number of bins in the histogram.
    density : bool
        If True, the histogram is normalized to form a probability density.
    """

    samples = [rv() for _ in range(num_samples)]
    plt.hist(samples, bins=bins, density=density)
    plt.show()


def mean(rv, max_iter=100000, tol=0.00001):
    """Numerically approximates the mean of a random variable."""

    max_small_change = 10

    total = 0
    avg = 0
    count = 0
    delta = tol + 1
    for i in range(1, max_iter):
        total += rv(i)
        new_avg = total / i
        delta = abs(new_avg - avg)
        avg = new_avg

        if delta <= tol:
            count += 1
            if count >= max_small_change:
                return avg

    print('Failed to converge')
