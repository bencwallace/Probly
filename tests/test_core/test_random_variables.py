from unittest import TestCase

import probly as pr


class TestArithmetic(TestCase):
    def setUp(self):
        self.seed = 11

        self.x = 333
        self.y = 112

        self.X = pr.const(self.x)
        self.Y = pr.const(self.y)

    def test_add(self):
        z = self.x + self.y
        Z = self.X + self.Y
        self.assertEqual(Z(self.seed), z)

    def test_sub(self):
        z = self.x - self.y
        Z = self.X - self.Y
        self.assertEqual(Z(self.seed), z)


class TestConditional(TestCase):
    def test_conditional(self):
        n = 10
        X = pr.Bin(n)
        Y = X + 1
        x = 5
        y = x + 1
        Z = Y.given(X == x)
        self.assertEqual(Z(), y)
