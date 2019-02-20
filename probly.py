"""Extensions to scipy.stats"""


class sampler():
    """
    A sampler.

    Basically a very general kind of random variable. Only capable of producing
    random quantities, whose explicit distribution is not necessarily known.
    """

    def __init__(self, f, *argv):
        self.argv = argv
        self.f = f

    def rvs(self):
        samples = [var.rvs() for var in self.argv]
        return self.f(*samples)


def Lift(f):
    """Lifts a function to the composition map between random variables."""

    def F(X):
        return sampler(f, X)

    return F
