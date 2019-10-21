import numpy as np
import probly as pr
import unittest
from unittest import TestCase
# from probly.core import Node
from probly.core import RandomVar


get_random = RandomVar._get_random
_max_seed = RandomVar._max_seed
msg_low_prob = 'Test fails with low probability. Try running tests again.'


class TestSimple(TestCase):
    """Tests for simple distributions."""

    def test_unif(self):
        # Almost surely passes
        a = 0
        b = 1

        X = pr.Unif(a, b)

        self.assertNotEqual(X(), X(), msg_low_prob)

    def test_ber(self):
        # I think this test is failing relatively frequently due to lack of
        # true independence
        p = 0.5

        X = pr.Ber(p)
        total1 = np.sum([X() for _ in range(100)])
        total2 = np.sum([X() for _ in range(100)])

        self.assertNotEqual(total1, total2, msg_low_prob)


class TestSimpleSeeded(TestCase):
    """Tests for simple distributions (seeded)."""

    def setUp(self):
        self.seed = 100

    def test_unif(self):
        a = -10
        b = 10

        X = pr.Unif(a, b)
        offset = get_random(X._id)
        np.random.seed((self.seed + offset) % _max_seed)
        x = np.random.uniform(a, b)

        self.assertEqual(X(self.seed), x)

    def test_ber(self):
        p = 0.8

        X = pr.Ber(p)
        offset = get_random(X._id)
        np.random.seed((self.seed + offset) % _max_seed)
        # x = np.random.choice(2, p=[1 - p, p])
        x = np.random.binomial(1, p)

        self.assertEqual(X(self.seed), x)


class TestDistr(TestCase):
    """Tests for a few other distributions."""
    def setUp(self):
        # Binomial
        self.n = 10
        self.p = 0.2

        # Beta
        self.alpha = 2.1
        self.beta = 3

        # Gamma
        self.shape = 1.5
        self.rate = 2.6
        self.scale = 3.1

        # Chi squared
        self.k = 11.1

    def test_bin(self):
        val = pr.Bin(self.n, self.p)()
        self.assertIsNotNone(val)

    def test_beta(self):
        val = pr.Beta(self.alpha, self.beta)()
        self.assertIsNotNone(val)

    def test_gamma_rate(self):
        val = pr.Gamma(self.shape, self.rate)()
        self.assertIsNotNone(val)

    def test_gamma_scale(self):
        val = pr.Gamma(self.shape, scale=self.scale)()
        self.assertIsNotNone(val)

    def test_gamma_both(self):
        val = pr.Gamma(self.shape, self.rate, self.scale)
        self.assertIsNotNone(val)

    def test_chisquared(self):
        val = pr.ChiSquared(self.k)
        self.assertIsNotNone(val)

    def test_exp(self):
        val = pr.Exp(self.rate)
        self.assertIsNotNone(val)


class TestScalar(TestCase):
    def setUp(self):
        self.seed = 90

    def test_add(self):
        a1 = -4
        b1 = 3

        a2 = -2.2
        b2 = 3.1

        X = pr.Unif(a1, b1)
        offset = get_random(X._id)
        np.random.seed((self.seed + offset) % _max_seed)
        x = np.random.uniform(a1, b1)

        Y = pr.Unif(a2, b2)
        offset = get_random(Y._id)
        np.random.seed((self.seed + offset) % _max_seed)
        y = np.random.uniform(a2, b2)

        Z = X + Y
        z = x + y

        self.assertEqual(Z(self.seed), z)

    def test_rmul(self):
        a = -8
        b = -1

        Y = pr.Unif(a, b)
        offset = get_random(Y._id)
        np.random.seed((self.seed + offset) % _max_seed)
        y = np.random.uniform(a, b)

        Z = 1.1 * Y
        z = 1.1 * y

        self.assertEqual(Z(self.seed), z)

    def test_neg(self):
        a = -9
        b = 10

        X = pr.Unif(a, b)
        offset = get_random(X._id)
        np.random.seed((self.seed + offset) % _max_seed)
        x = np.random.uniform(a, b)

        Z = -X

        self.assertEqual(Z(self.seed), -x)


class TestCopy(TestCase):
    def setUp(self):
        self.seed = 1

    def test_indep_copy(self):
        X = pr.Unif(-1, 1)
        Y = X.copy()

        self.assertNotEqual(X(self.seed), Y(self.seed))

    def test_dependent_copy(self):
        X = pr.Unif(-1, 1)
        Z = pr.array([X, X])

        self.assertEqual(Z(self.seed)[0], Z(self.seed)[1])


class TestArray(TestCase):
    def setUp(self):
        self.seed = 99

    # Isn't actually doing any multiplication...
    def test_matmul(self):
        X = pr.Unif(-1, 1)
        Y = pr.Unif(-1, 1)
        Z = pr.array(([X, Y], np.array([Y, 1])))
        # W = np.dot(Z, Z)

        offset = get_random(X._id)
        np.random.seed((self.seed + offset) % _max_seed)
        x = np.random.uniform(-1, 1)

        offset = get_random(Y._id)
        np.random.seed((self.seed + offset) % _max_seed)
        y = np.random.uniform(-1, 1)

        # self.assertAlmostEqual(np.linalg.det(W(self.seed)), (y**2-x)**2)
        self.assertAlmostEqual(np.linalg.det(Z(self.seed)), x - y ** 2)

    def test_getitem(self):
        # Bug: doesn't work with silly nestings like [[X]]
        X = pr.Unif(-1, 1)
        Z = pr.array([[X, 1], [1, 1]])

        offset = get_random(X._id)
        np.random.seed((self.seed + offset) % _max_seed)
        x = np.random.uniform(-1, 1)

        self.assertEqual(Z[0, 0](self.seed), x)


if __name__ == '__main__':
    unittest.main()
