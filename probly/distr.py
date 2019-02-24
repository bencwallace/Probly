"""
Random variables for common distributions.

Subclassing instructions:

    To define a random variable class with a desired distribution, create a
    subclass of `Distr` with a single method `sampler(self, seed)`, which
    samples from the desired distribution. Seeding according to `seed` is taken
    care of by the `rv` class. Initialization of the subclass should take a
    list of parameters, which can be recovered in the `sampler` method from
    `self.params`.

Subclassing example:

    import numpy as np
    from probly.distr import Distr, Ber

    class Human(object):
        def __init__(self, gender, height, weight):
            self.gender = gender
            self.height = height
            self.weight = weight

        def BMI(self):
            return self.weight / self.height ** 2

    class randomHuman(Distr):
        def sampler(self, seed=None):
            female_stats, male_stats = self.params

            gender = {0: 'F', 1: 'M'}[Ber(0.5)(seed)]
            if gender == 0:
                height_mean, weight_mean, cov = female_stats
            else:
                height_mean, weight_mean, cov = male_stats

            means = [height_mean, weight_mean]
            height, weight = np.random.multivariate_normal(means, cov)

            return Human(gender, height, weight)

    f_cov = np.array([[80, 5], [5, 99]])
    f_stats = [160, 65, f_cov]

    m_cov = np.array([[70, 4], [4, 110]])
    m_stats = [180, 75, m_cov]

    H = randomHuman(f_stats, m_stats)
"""

import numpy as np
from . import rv


class Distr(rv):
    def __init__(self, *args):
        self.params = args
        super().__init__()


# To do: allow a, b random
class Unif(Distr):
    def sampler(self, seed=None):
        a, b = self.params
        return np.random.uniform(a, b)


class Ber(Distr):
    def sampler(self, seed=None):
        p, = self.params
        return np.random.choice(2, p=[1 - p, p])
