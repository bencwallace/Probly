import numpy as np

from unittest import TestCase

import probly as pr


class TestRandomArrays(TestCase):
    def test_getitem(self):
        seed = 12000
        N = pr.Normal()
        X = pr.RandomArray([N])
        M = X[0]
        self.assertEqual(M(seed), N(seed))

    def test_array(self):
        X = pr.array(pr.const(10), 10)
        Y = np.sum(X)
        self.assertEqual(Y(), 100)
