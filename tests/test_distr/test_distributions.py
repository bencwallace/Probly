import itertools
import numpy as np

from unittest import TestCase

import probly as pr

current_id = itertools.count(start=1)


class TestDistributions(TestCase):
    max_seed = 2 ** 32 - 1
    current_id = itertools.count(start=1)

    def setUp(self):
        self.user_seed = 666

    def seed(self, rv):
        return (self.user_seed + rv._offset) % self.max_seed


class TestRandomDistributions(TestDistributions):
    def test_ber_of_unif(self):
        n = 10
        p = 0.6
        U = pr.Unif(p)
        X = pr.Bin(n, U)
        np.random.seed(self.seed(U))
        u = np.random.uniform(p)
        np.random.seed(self.user_seed)
        x = np.random.binomial(n, u)
        self.assertEqual(X(self.user_seed), x)
