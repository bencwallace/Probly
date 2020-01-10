import itertools
import numpy as np

from unittest import TestCase

import probly as pr

from .test_distributions import TestDistributions


class TestContinuous(TestDistributions):
    def test_unif(self):
        a = -100
        b = 100
        X = pr.Unif(a, b)
        np.random.seed(self.seed(X))
        x = np.random.uniform(a, b)
        self.assertEqual(X(self.user_seed), x)
