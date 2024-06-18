from typing import List, Type, Generic, TypeVar

from src.entities.neuron import Node
from src.patterns.observer import Observer

NodeT = TypeVar('NodeT', bound=Node)


class NodeList(Generic[NodeT], Observer):
    """
    Immutable list of nodes.
    """

    nodes: List[Node]
    _node_type: Type

    def __init__(self, nodes: List[Node], node_type: Type) -> None:
        self.nodes = nodes
        self._node_type = node_type

    def get_node_type(self) -> Type:
        return self._node_type

    def notify(self, cmd: str, node: Node):
        matches_type = isinstance(node, self._node_type)

        if not matches_type:
            raise ValueError(f'Expected one of {str(self._node_type)}, '
                             f'received {type(node)}')

        if cmd == 'remove':
            self.nodes.remove(node)
        elif cmd == 'add':
            self.nodes.append(node)
        else:
            raise ValueError(f'"{cmd}" is not a recognised node list command.')

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, item):
        return self.nodes[item]

    def __len__(self):
        return len(self.nodes)