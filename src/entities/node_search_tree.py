from typing import Union, Tuple, List

import numpy as np
from sklearn.neighbors import KDTree

from src.entities.cortex_node import CortexNode
from src.entities.neuron import Coords
from src.entities.node_list import NodeList


class NodeSearchTree:
    """
    Implements a search tree for nodes.
    """

    def __init__(self, nodes: List[CortexNode] | NodeList[CortexNode]):
        """
        Construct a tree for fast searching of the given nodes by their
        coordinates.
        :param nodes: A list of nodes.
        """
        self.nodes = nodes
        rank = len(nodes[0].get_coords())
        array = np.zeros((len(nodes), rank), dtype=np.double)
        for i, node in enumerate(nodes):
            array[i, :] = node.get_coords()
        self.tree = KDTree(array)

    def get_k_nearest(self, node: Union[CortexNode, Coords], k: int) -> \
            List[Tuple[float, CortexNode]]:

        if isinstance(node, CortexNode):
            coords = node.get_coords()
        else:
            coords = node
        array = np.array(coords, dtype=np.double)
        array.shape = [1, len(coords)]
        d, ind = self.tree.query(array, return_distance=True, k=k)

        rtrn = []
        for i in range(k):
            rtrn.append((d[0, i], self.nodes[ind[0, i]]))
        return rtrn

    def get_in_radius(self, node: Union[CortexNode, Coords], dist: float) -> \
            List[Tuple[float, CortexNode]]:
        """
        Find all nodes within <dist> distance of <node>.
        :param dist:
        :param node:
        :return:
        """
        if isinstance(node, CortexNode):
            coords = node.get_coords()
        else:
            coords = node
        array = np.array(coords, dtype=np.double)
        array.shape = (1, len(node.get_coords()))
        ind, d = self.tree.query_radius(array, return_distance=True, r=dist,
                                        sort_results=True)

        rtrn = []
        for i in range(len(d[0])):
            rtrn.append((d[0][i], self.nodes[ind[0][i]]))
        return rtrn
