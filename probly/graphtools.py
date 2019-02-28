import networkx as nx


# Initialize global dependency graph
_graph = nx.MultiDiGraph()


def nodes():
    return [str(n) for n in _graph.nodes]


def edges():
    return [(str(x), str(y)) for (x, y, z) in _graph.edges]
