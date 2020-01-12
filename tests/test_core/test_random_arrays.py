import numpy as np

from unittest import TestCase

import probly as pr
import probly.lib.utils


class TestRandomArrays(TestCase):
    def test_getitem(self):
        seed = 12000
        N = pr.Normal()
        X = probly.lib.utils.array([N])
        M = X[0]
        self.assertEqual(M(seed), N(seed))

    def test_array(self):
        X = probly.lib.utils.iid(pr.const(10), 10)
        Y = np.sum(X)
        self.assertEqual(Y(), 100)
