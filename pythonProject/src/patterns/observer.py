from abc import abstractmethod, ABC
from typing import Any


class Observer(ABC):

    @abstractmethod
    def notify(self, cmd: str, obj: Any):
        pass
