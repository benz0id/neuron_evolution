from typing import List, Union

from src.presenters.matrix_presenter import PresenterSubject, Presenter


class EndOfSimulation(Exception):
    def __init__(self, epoch):
        self.epoch = epoch
    def __str__(self):
        return f'Simulation terminated at epoch {self.epoch}'

class EpochCounter(PresenterSubject):
    i: int
    _termination_epoch: Union[int, None]

    def __init__(self, presenters: List[Presenter],
                 termination_epoch: Union[int, None]):
        super().__init__(presenters)
        self.i = 0
        self._termination_epoch = termination_epoch

    def iterate(self):
        self.update('epoch', None)
        self.i += 1
        if self._termination_epoch is not None and \
            self.i > self._termination_epoch:
            raise EndOfSimulation(self.i)


    def get_epoch(self) -> int:
        return self.i
