class Condition(object):
    def __init__(self, Y=None, p=None):
        if p:
            self.requirements = {Y: [p]}
        else:
            self.requirements = {}

    def require(self, rv, p):
        if self.requirements[rv]:
            self.requirements[rv].append(p)
        else:
            self.requirements[rv] = [p]

    def check(self, seed):
        for rv, ps in self.requirements.items():
            for p in ps:
                if not p(rv(seed)):
                    return False
        return True
