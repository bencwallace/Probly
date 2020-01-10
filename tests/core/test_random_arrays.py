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
        # make random array with array and test with getitem
        pass

    def test_wigner(self):
        pass

    def test_wishart(self):
        pass
