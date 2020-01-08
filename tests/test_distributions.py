import itertools
import numpy as np

from numpy.testing import assert_array_equal
from unittest import TestCase

import probly as pr


max_seed = 2 ** 32 - 1
current_id = itertools.count(start=1)


class TestDistribution(TestCase):
    def setUp(self):
        self.user_seed = 666
        rv_id = next(current_id)
        np.random.seed(rv_id)
        current_offset = np.random.randint(max_seed)
        self.seed = (self.user_seed + current_offset) % max_seed

    def test_choice(self):
        a = -100
        b = 100
        X = pr.RandInt(a, b)
        np.random.seed(self.seed)
        x = np.random.randint(a, b)
        self.assertEqual(X(self.user_seed), x)

    def test_multinomial(self):
        n = 100
        pvals = [1 / n] * n
        X = pr.Multinomial(n, pvals)
        np.random.seed(self.seed)
        x = np.random.multinomial(n, pvals)
        assert_array_equal(X(self.user_seed), x)

    def test_unif(self):
        a = -100
        b = 100
        X = pr.Unif(a, b)
        np.random.seed(self.seed)
        x = np.random.uniform(a, b)
        self.assertEqual(X(self.user_seed), x)
