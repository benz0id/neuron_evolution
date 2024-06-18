import abc

from src.entities.node import Node, Coords


class CortexNode(Node):

    """
    A node with a position in the cortex.
    """

    @abc.abstractmethod
    def get_coords(self) -> Coords:
        pass

    @abc.abstractmethod
    def get_next_coords(self) -> Coords:
        pass

    @abc.abstractmethod
    def move_to(self, coords: Coords) -> None:
        pass

