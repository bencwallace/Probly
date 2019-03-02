"""
Core objects and methods
========================

This module defines the global dependency graph, its `Node` objects, and its
`Root` object.
"""

import numpy as np

import copy
import itertools

import networkx as nx


def get_offset(_id):
    np.random.seed(_id)
    # Convert to int to avoid overflow error
    return int(np.random.get_state()[1][1])


class Node(object):
    """
    A node in the global dependency graph.

    The dependency graph is a labeled directed multigraph implemented with the
    NetworkX package. The nodes of the graph are `Node` objects and are labeled
    by a callable object, which we refer to as its "call method". The primary
    function of a `Node` object is, when
    called, to recursively combine the outputs of its parents in the
    dependency graph by feeding them as arguments to its call method. The order
    in which the outputs are mapped to arguments is determined by the edge
    labels.

    Parameters
    ----------
    call_method : callable
    parents : list
        List of `Node` objects

    Note
    ----
    `Node` objects are not meant to be instantiated directly.
    """

    # Track random variables for independence
    _last_id = itertools.count()

    # Initialize dependency graph
    _graph = nx.MultiDiGraph()

    # NumPy max seed
    _max_seed = 2 ** 32 - 1

    # Core magic methods
    def __new__(cls, call_method=None, *parents):
        obj = super().__new__(cls)

        edges = [(cls._cast(var), obj, {'index': i})
                 for i, var in enumerate(parents)]
        cls._graph.add_node(obj, call_method=call_method)
        cls._graph.add_edges_from(edges)

        # Can't rely on init constructor, which gets overloaded
        obj._id = next(cls._last_id)
        obj._offset = get_offset(obj._id)

        return obj

    def __call__(self, seed=None):
        """
        Produces a random sample from the desired distribution.

        Parameters
        ----------
        seed : int, optional
            A seed value. If not specified, a random seed will be used.
        """

        call_method = self._graph.nodes[self]['call_method']

        seed = RNG(seed)
        parents = self.parents()
        samples = [parents[i](seed)
                   for i in range(len(parents))]

        return call_method(*samples)

    # Representation magic
    def __str__(self):
        return 'RV {}'.format(self._id)

    # Helper methods
    def parents(self):
        """Returns the list of parents in the dependency graph."""

        if self not in self._graph:
            return []
        else:
            # Convert to list for re-use
            unordered = list(self._graph.predecessors(self))
            if len(unordered) == 0:
                return []

            data = [self._graph.get_edge_data(p, self) for p in unordered]

            # Create {index: parent} dictionary
            dictionary = {}
            for i in range(len(unordered)):
                indices = [d.values() for d in data[i].values()]
                for j in range(len(indices)):
                    dictionary[data[i][j]['index']] = unordered[i]

            ordered = [dictionary[i] for i in range(len(dictionary))]
            return ordered

    def copy(self):
        """Returns an independent random variable of the same distribution."""

        return copy.copy(self)

    def __copy__(self):
        # Returns a seed-shifted (independent) version of `self`

        # Save id
        next_id = next(self._last_id)
        next_offset = get_offset(next_id)

        # Construct shifted copy
        def shifted_call_method(seed=None):
            diff = self._offset - next_offset
            return self((RNG(seed) - diff) % self._max_seed)
        Copy = self.__new__(type(self), shifted_call_method, RNG)

        # indep_call_method = make_indep(self, next_id)
        # Copy = self.__new__(type(self), call_method, RNG)

        # Copy data that may exist in subclass. Old id also gets copied over
        for key, val in self.__dict__.items():
            setattr(Copy, key, val)

        # Reset id to new id
        Copy._id = next_id
        Copy._offset = next_offset

        return Copy


class Root(Node):
    """
    A root node of the dependency graph. Acts as a random number generator.
    """

    def __new__(cls, *args, **kwargs):
        return super(Node, cls).__new__(cls, *args, **kwargs)

    def __call__(self, seed=None):
        """
        Generate a random seed. If a seed is provided, returns it unchanged.
        """
        np.random.seed(seed)
        return np.random.get_state()[1][0]


# Make random number generator root
RNG = Root()
