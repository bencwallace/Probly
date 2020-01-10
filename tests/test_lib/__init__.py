from unittest import TestCase

import probly as pr


class TestUtils(TestCase):
    def setUp(self):
        self.c = 192
        self.C = pr.const(self.c)

    def test_const(self):
        self.assertEqual(self.C(), self.c)

    def test_lift(self):
        @pr.lift
        def f(x):
            return x ** 2
        X = f(self.C)
        x = f(self.c)
        self.assertEqual(X(), x)
