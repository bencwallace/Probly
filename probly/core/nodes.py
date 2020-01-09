class Node(object):
    def __init__(self, op, *parents):
        self.parents = parents

        if not callable(op):
            # Treat op as constant
            self.op = lambda *x: op
        else:
            self.op = op

    def __call__(self, *args):
        if not self.parents:
            # Let root act directly on args
            out = self.op(*args)
        else:
            inputs = (p(*args) for p in self.parents)
            out = self.op(*inputs)

        # For length 1 tuples
        if hasattr(out, '__len__') and len(out) == 1:
            out = out[0]

        return out
