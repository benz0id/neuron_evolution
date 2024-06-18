from itertools import count
from typing import List, Type, Dict

from src.entities.neuron import Neuron, Node, CortexNode
from src.entities.node_list import NodeList
from src.managers.manager import Manager

NODE_TYPES = [Node, CortexNode, Neuron]

class NodeManager(Manager):

    """
    Stores neurons for shared access. Manages the addition and deletion of
    nodes across managers.

    If a manager stores a node, it must be added to its respective list of
    managers within a NodeManager.
    """

    _nodes = List[Node]
    _node_lists = Dict[Type, NodeList]
    _managers = List[Manager]
    _node_id_counter: count

    def __init__(self) -> None:
        super().__init__(Node)
        self.population = []
        self._nodes = []
        self._node_lists = {}
        self._managers = []


        for t in NODE_TYPES:
            self._node_lists[t] = NodeList([], t)

        self._node_id_counter = count()

    def get_next_id(self) -> int:
        return next(self._node_id_counter)

    def get_neuron_count(self):
        return len(self.get_all(Neuron))

    def get_all(self, t: Type[Node]) -> NodeList:
        """
        Get a nodelist of all nodes with type <t>
        :param t:
        :return:
        """
        if t not in self._node_lists:
            raise ValueError(f'No nodes of type {t} have been added to the '
                             f'list.')
        return self._node_lists[t]

    def add_manager(self, manager: Manager):
        self._managers.append(manager)

    def add_node(self, node: Node) -> None:
        self._nodes.append(node)

        t = type(node)

        for ot in self._node_lists:
            if issubclass(t, ot):
                self._node_lists[ot].notify('add', node)

        for manager in self._managers:
            for ot in manager.get_node_types():
                if issubclass(t, ot):
                    manager.notify('add', node)
                    break

    def remove_node(self, node: Node) -> None:
        self._nodes.remove(node)

        t = type(node)

        for ot in self._node_lists:
            if issubclass(t, ot):
                self._node_lists[ot].notify('remove', node)

        for manager in self._managers:
            for ot in manager.get_node_types():
                if issubclass(t, ot):
                    manager.notify('remove', node)
                    break

        node.disconnect()

    def _add(self, node: Node) -> None:
        pass

    def _remove(self, node: Node) -> None:
        pass




