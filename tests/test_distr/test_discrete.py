import itertools
import numpy as np

from numpy.testing import assert_array_equal
from unittest import TestCase

import probly as pr

from .test_distributions import TestDistributions


class TestDiscrete(TestDistributions):
    def test_rand_int(self):
        a = -100
        b = 100
        X = pr.RandInt(a, b)
        np.random.seed(self.seed(X))
        x = np.random.randint(a, b)
        self.assertEqual(X(self.user_seed), x)

    def test_multinomial(self):
        n = 100
        pvals = [1 / n] * n
        X = pr.Multinomial(n, pvals)
        np.random.seed(self.seed(X))
        x = np.random.multinomial(n, pvals)
        assert_array_equal(X(self.user_seed), x)
