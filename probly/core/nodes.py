class Node(object):
    """
    A node in a computational graph.

    :param op: callable
        The operation represented by this node.
    :param parents: Node, optional
        The input nodes.
    """

    def __init__(self, op=None, *parents):
        self.parents = parents
        if op is not None and not callable(op):
            self.op = lambda _: op
        else:
            self.op = op

    def __call__(self, *args):
        """
        Evaluates the node.

        If the node is a root node (i.e. has no parents), it's operation is evaluated
        directly on the arguments.
        """
        if self.op is None:
            return self._default_op(*args)
        elif not self.parents:
            # Let root act directly on args
            return self.op(*args)
        else:
            inputs = (p(*args) for p in self.parents)
            return self.op(*inputs)

    def _default_op(self, *args):
        return args
