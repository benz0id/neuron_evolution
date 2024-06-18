import abc

import numpy as np

Coords = np.array
class Node(abc.ABC):
    _creation_epoch: int
    _node_id: int

    def __init__(self, creation_epoch: int, node_id: int):
        self._creation_epoch = creation_epoch
        self._node_id = node_id

    @abc.abstractmethod
    def get_state(self) -> int:
        pass

    @abc.abstractmethod
    def add_output(self, output):
        pass

    @abc.abstractmethod
    def remove_output(self, output):
        pass

    @abc.abstractmethod
    def add_input(self, input):
        pass

    @abc.abstractmethod
    def remove_input(self, input):
        pass

    @abc.abstractmethod
    def is_excited(self) -> bool:
        pass

    def get_id(self) -> int:
        return self._node_id

    def get_creation_epoch(self) -> int:
        return self._creation_epoch

    @abc.abstractmethod
    def add_reward(self, reward) -> None:
        pass

    @abc.abstractmethod
    def add_cogni(self, amount):
        pass

    @abc.abstractmethod
    def sub_cogni(self, amount):
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass
