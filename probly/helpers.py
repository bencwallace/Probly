"""
Methods for analyzing random variables.
"""

import matplotlib.pyplot as plt


def hist(rv, num_samples, bins=None, density=True):
    """
    Plots a histogram from samples of a random variable.

    Parameters
    ----------
    rv : RandomVar
        A random variable.
    num_samples : int
        The number of samples to draw from `rv`.
    bins : int or sequence, optional
        Specifies the bins in the histogram.
    density : bool, optional
        If True, the histogram is normalized to form a probability density.
    """

    samples = [rv() for _ in range(num_samples)]
    plt.hist(samples, bins=bins, density=density)
    plt.show()
