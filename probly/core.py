"""
Core objects and methods
========================

This module defines the global dependency graph, its `Node` objects, and its
`Root` object.
"""

import math
import numpy as np

import copy
import itertools

import networkx as nx

import os


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
    parents : list of `Node` objects

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

        obj._id = next(cls._last_id)

        edges = [(cls._cast(var), obj, {'index': i})
                 for i, var in enumerate(parents)]
        cls._graph.add_node(obj, call_method=call_method)
        cls._graph.add_edges_from(edges)

        return obj

    def __call__(self, seed=None):
        """
        Produces a random sample from the desired distribution.

        Parameters
        ----------
        seed : int, optional
            A seed value. If not specified, a random seed will be used.
        """

        seed = root(seed)
        parents = self.parents()

        samples = [parents[i](seed)
                   for i in range(len(parents))]

        call_method = self._graph.nodes[self]['call_method']
        if call_method is 'sampler':
            # kluge (only happens once but still kluge)
            def seeded_sampler(seed):
                return self._sampler((seed + self._id) % self._max_seed)
            call_method = seeded_sampler
            nx.set_node_attributes(self._graph,
                                   {self: {'call_method': call_method}})
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
        call_method = self._graph.nodes[self]['call_method']
        parents = self.parents()

        Copy = self.__new__(type(self), call_method, *parents)
        _id = Copy._id

        for key, val in self.__dict__.items():
            setattr(Copy, key, val)

        # Do not copy id
        Copy._id = _id

        return Copy


class Root(Node):
    def __new__(cls, *args, **kwargs):
        return super(Node, cls).__new__(cls, *args, **kwargs)

    def __call__(self, seed=None):
        """
        Generate a random seed. If a seed is provided, returns it unchanged.

        Based on the Python implementation. A consistent approach to generating
        re-usable random seeds is needed in order to implement dependency.
        """

        if seed is not None:
            return seed

        try:
            max_bytes = math.ceil(np.log2(self._max_seed) / 8)
            seed = int.from_bytes(os.urandom(max_bytes), 'big')
        except NotImplementedError:
            raise NotImplementedError('Seed from time not implemented.')

        return seed


root = Root()
