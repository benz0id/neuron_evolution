from dataclasses import dataclass
from typing import Type

import numpy as np


@dataclass
class MatrixFeature:

    coords: np.array
    next_coords: np.array
    id: int
    group: int
    entity_type: Type

    def __hash__(self):
        r = self.id
        return r

    def __eq__(self, other):
        return self.id == other.id
