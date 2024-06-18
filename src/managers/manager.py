from abc import abstractmethod
from typing import Type, Union, List

from src.entities.neuron import Node
from src.patterns.observer import Observer


class Manager(Observer):
    """
    A manager that may store one or more type of Node.
    """

    # The types of node that this manager stores
    _node_types: List[Type]

    def __init__(self, node_types: Union[Type[Node], List[Type[Node]]]) -> None:
        """
        :param node_types: The types of nodes that this manager can store.
        """
        if isinstance(node_types, List):
            self._node_types = node_types
        else:
            self._node_types = [node_types]

    def get_node_types(self) -> List[Type]:
        """
        :return: The types of nodes that this manager can store.
        """
        return self._node_types

    def notify(self, cmd: str, node: Node):
        """
        Either add or remove the given node from this Manager's data structures.
        :param cmd: Either "add" or "remove"
        :param node: The node to be added or removed.
        :return:
        """
        matches_type = any(isinstance(node, t) for t in self._node_types)

        if not matches_type:
            raise ValueError(f'Expected one of {str(self._node_types)}, '
                             f'received {type(node)}')

        if cmd == 'remove':
            self._remove(node)
        elif cmd == 'add':
            self._add(node)
        else:
            raise ValueError(f'"{cmd}" is not a recognised manager command.')

    @abstractmethod
    def _remove(self, node: Node) -> None:
        """
        :param node: Node to be removed from this Manager's data structures.
        :return:
        """
        pass

    @abstractmethod
    def _add(self, neuron: Node) -> None:
        """
         Node to be added to this Manager's data structures.
        :param neuron:
        :return:
        """
        pass
