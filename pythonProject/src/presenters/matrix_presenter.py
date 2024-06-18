import sys
from abc import abstractmethod
from collections import deque
from copy import copy
from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import Dict, Set, List, Union, Tuple, Callable

import numpy as np
import serial
from serial.serialutil import SerialException

from src.entities.cortex_node import CortexNode
from src.entities.neuron import Neuron
from src.managers.manager import Manager
from src.presenters.matrix_feature import MatrixFeature


def add(dict, key, val):
    if key in dict:
        dict[key] += val
    else:
        dict[key] = val

def sub(dict, key, val):
    if key in dict:
        dict[key] -= val
    else:
        raise ValueError()


class Presenter(Manager):

    to_exec: deque[Tuple[Callable, int]]
    feature_cache: List[int | MatrixFeature]

    def __init__(self):
        super().__init__(CortexNode)
        self.to_exec = deque()
        self.feature_cache = []

    @abstractmethod
    def _move(self, feature: MatrixFeature):
        pass

    def notify(self, cmd: str, node: Union[CortexNode, None]):
        """
        Either add or remove the given feature from this Manager's data structures.
        :param cmd: Either "add", "remove", or "move"
        :param node: The feature to be added or removed.
        :return:
        """
        matches_type = any(isinstance(node, t) for t in self._node_types)
        if not matches_type and node is not None:
            raise ValueError(f'Expected one of {str(self._node_types)}, '
                             f'received {type(node)}')
        
        if isinstance(node, Neuron):
            feature = MatrixFeature(coords=copy(node.get_coords()),
                                    next_coords=copy(node.get_next_coords()),
                                    id=node.get_id(),
                                    group=node.get_species_id(),
                                    entity_type=Neuron)
            fid = feature.id
            while fid >= len(self.feature_cache):
                self.feature_cache.append(0)
            self.feature_cache[fid] = feature
        else:
            feature = None

        if feature is None:
            assert cmd == 'epoch'
            self._epoch()
            while len(self.to_exec) > 0:
                method, fid = self.to_exec.pop()
                feature = self.feature_cache[fid]
                method(feature)

        elif cmd == 'death':
            self.to_exec.appendleft((self._remove, feature.id))
        elif cmd == 'born':
            self.to_exec.appendleft((self._add, feature.id))


        elif cmd == 'move':
            self.to_exec.appendleft((self._move, feature.id))


        elif cmd == 'mate':
            self.to_exec.appendleft((self._mate, feature.id))
        elif cmd == 'clone':
            self.to_exec.appendleft((self._clone, feature.id))
        elif cmd == 'fire':
            self.to_exec.appendleft((self._fire, feature.id))
        elif cmd == 'reward':
            self.to_exec.appendleft((self._reward, feature.id))


        else:
            raise ValueError(f'"{cmd}" is not a recognised manager command.')

    def _epoch(self):
        pass

    def _reward(self, feature: MatrixFeature):
        pass

    def _mate(self, feature: MatrixFeature):
        pass

    def _clone(self, feature: MatrixFeature):
        pass

    def _fire(self, feature: MatrixFeature):
        pass


class PresenterSubject:

    _presenters: List[Presenter]

    def __init__(self, presenters: List[Presenter]):
        self._presenters = presenters

    def update(self, cmd: str, feature: Union[CortexNode, None]):
        for presenter in self._presenters:
            presenter.notify(cmd, feature)

@dataclass
class Marker:
    kind: str
    duration: int
    coords: np.array
    mid: int

    def __hash__(self) -> int:
        return self.mid

class MatrixPresenter(Presenter):
    feature_layer: np.array
    marker_layer: np.array
    sector_layer: np.array

    species_to_colour: Dict[int, int]
    species_to_count: Dict[int, int]
    neuron_to_species: Dict[MatrixFeature, int]

    available_colours: Set[int]

    markers: Set[Marker]
    mid: int = -1

    width: int
    height: int

    fps: float
    update_every: int

    port: str

    verbose: bool

    event_to_palette = {
        'death': 1,
        'mate': 2,
        'clone': 5,
        'fire': 6,
        'born': 7,
        'reward': 8,
        'sector_boundary': 9
    }

    event_to_marker_duration = {
        'death': 1,
        'mate': 1,
        'clone': 1,
        'fire': 1,
        'born': 1,
        'reward': 1
    }

    def __init__(self, width: int, height: int,
                 fps: int, update_every: int,
                 max_colours: int,
                 num_reserved: int,
                 verbose: bool,
                 port: str):
        super().__init__()
        self.port = port
        self.verbose = verbose
        self.update_every = update_every
        self.width = width
        self.height = height
        self.feature_layer = np.zeros((width, height), dtype=np.int32)
        self.marker_layer = np.zeros((width, height), dtype=np.int32)
        self.sector_layer = np.zeros((width, height), dtype=np.int32)

        self.species_to_colour = {}
        self.species_to_count = {}

        self.neuron_to_species = {}

        self.markers = set()

        self.available_colours = set([i for i in range(num_reserved,
                                                       max_colours)])

        # Avoid termination signals.
        if not num_reserved > 4:
            self.available_colours.remove(4)
            self.available_colours.remove(3)
        self.fps = fps
        self.thread = Thread(target=self.update_matrix_display)
        self.thread.start()

    def get_mid(self) -> int:
        self.mid += 1
        return self.mid

    def draw_sectors(self, sector_size: int):
        for i in range(self.width):
            for j in range(self.height):
                if (i + 1) % sector_size == 0 or (j + 1) % sector_size == 0:
                    self.sector_layer[i, j] = \
                        self.event_to_palette['sector_boundary']
        pass


    def update_matrix_display(self):

        try:
            s = serial.Serial(self.port)
        except SerialException:
            print('Connection to LED panel Failed.', file=sys.stderr)
            return

        delay = 1 / self.fps

        c = 0
        while True:
            msg = []
            for j in range(self.height - 1, -1, -1):
                for i in range(self.width):

                    if self.marker_layer[i, j]:
                        byte = self.marker_layer[i, j]
                    elif self.feature_layer[i, j]:
                        byte = self.feature_layer[i, j]
                    elif self.sector_layer[i, j]:
                        byte = self.sector_layer[i, j]
                    else:
                        byte = 0

                    msg.append(int(byte).to_bytes(1))
            s.write(b''.join(msg))

            if self.verbose:
                print(msg)

            received = s.readline().decode()

            if self.verbose:
                print(received)

            sleep(delay)
            c += 1
            if c == self.update_every:
                c = 0
                self.refresh()

    def refresh(self):
        # Clear extinct species.
        all_species = list(self.species_to_count.keys())
        for species in all_species:
            if self.species_to_count[species] == 0:
                del self.species_to_count[species]
                colour = self.species_to_colour[species]
                del self.species_to_colour[species]
                self.available_colours.add(colour)

    def remove_feature(self, feature: MatrixFeature, last: bool = True):
        if last:
            coords = feature.coords
        else:
            coords = feature.next_coords
        self.feature_layer[coords[0], coords[1]] = 0
        if isinstance(feature, MatrixFeature):
            species = self.neuron_to_species[feature]
            sub(self.species_to_count, species, 1)

    def add_feature(self, feature: MatrixFeature):
        if isinstance(feature, MatrixFeature):
            species = feature.group
            if species in self.species_to_count:
                add(self.species_to_count, species, 1)
            else:
                add(self.species_to_count, species, 1)
                if len(self.available_colours) == 0:
                    self.refresh()
                colour = min(list(self.available_colours))
                self.species_to_colour[species] = colour
                self.available_colours.remove(colour)
            colour = self.species_to_colour[species]
            coords = feature.next_coords
            self.neuron_to_species[feature] = species
            self.feature_layer[coords[0], coords[1]] = colour

    def _add(self, neuron: MatrixFeature) -> None:
        self.handle_event('born', neuron)
        self.add_feature(neuron)

    def _remove(self, neuron: MatrixFeature) -> None:
        self.handle_event('death', neuron)
        self.remove_feature(neuron, last=False)

    def _move(self, neuron: MatrixFeature) -> None:
        self.remove_feature(neuron)
        self.add_feature(neuron)


    def handle_event(self, event: str, feature: MatrixFeature):
        duration = self.event_to_marker_duration[event]
        pal = self.event_to_palette[event]
        coords = feature.next_coords

        self.markers.add(Marker(
            event, duration, coords, self.get_mid()
        ))

        self.marker_layer[coords[0], coords[1]] = pal

    def _epoch(self):
        to_rem = []
        for marker in self.markers:
            marker.duration -= 1
            if marker.duration <= 0:
                to_rem.append(marker)
        for marker in to_rem:
            self.markers.remove(marker)
            coords = marker.coords
            self.marker_layer[coords[0], coords[1]] = 0

    def _reward(self, feature: MatrixFeature):
        self.handle_event('reward', feature)

    def _mate(self, feature: MatrixFeature):
        self.handle_event('mate', feature)

    def _clone(self, feature: MatrixFeature):
        self.handle_event('clone', feature)

    def _fire(self, feature: MatrixFeature):
        self.handle_event('fire', feature)