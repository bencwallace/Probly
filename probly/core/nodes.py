class Node(object):
    """
    A node in a computational graph.

    :param op: callable
        The operation represented by this node.
    :param parents: Node, optional
        The input nodes.
    """

    def __init__(self, op, *parents):
        self.parents = parents

        if not callable(op):
            # Treat op as constant
            self.op = lambda *x: op
        else:
            self.op = op

    def __call__(self, *args):
        """
        Evaluates the node.

        If the node is a root node (i.e. has no parents), it's operation is evaluated
        directly on the arguments.
        """
        if not self.parents:
            # Let root act directly on args
            out = self.op(*args)
        else:
            inputs = (p(*args) for p in self.parents)
            out = self.op(*inputs)

        return out
