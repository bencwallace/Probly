import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from probly import Lift


def chisq(num_samples, bins=20):
    """Generate a chi-squared distribution from a normal distribution."""

    def square(x):
        return x ** 2
    square = Lift(square)

    n = stats.norm()
    chisq = square(n)

    samples = [chisq.rvs() for _ in range(num_samples)]
    plt.hist(samples, bins=bins)
    plt.show()


def wish_eig(num_samples, bins=20):
    """Generate the largest eigenvalue of a Wishart matrix."""

    def largest_lambda(w):
        return np.max(np.linalg.eig(w)[0])
    largest_lambda = Lift(largest_lambda)

    W = stats.wishart(scale=np.eye(2))
    eig = largest_lambda(W)

    samples = [eig.rvs() for _ in range(num_samples)]
    plt.hist(samples, bins=bins)
    plt.show()
