import random
from copy import copy
from itertools import product
from typing import List, Dict, Tuple, Union

import numpy as np

from src.entities.neuron import CortexNode, Coords
from src.entities.node_list import NodeList
from src.entities.node_search_tree import NodeSearchTree
from src.managers.manager import Manager
from src.presenters.matrix_presenter import Presenter, PresenterSubject


class CortexManager(Manager, PresenterSubject):
    """
    Responsible for managing positioning related attributes of nodes and
    maintaining useful data structures for managing the cortex.

    shape: The dimensionality of the cortex.
    rank: The number of dimensions in the cortex.

    sector_size: Cortex is divided into prod(shape // sector_size) hypercubes.
    sector_shape: The shape of the sector-space.

    nodes: The nodes in the cortex.

    sectors: Maps the origins of each sector to all the nodes in that
    sector.

    """

    shape: List[int]
    rank: int

    sector_shape: List[int]
    sector_size: int
    node_density_contribution: float

    conversion_list: List

    nodes: NodeList[CortexNode]

    sector_densities: List[float]

    sector_surroundings: List[List[float]]

    euclidian_scaling_factor: float

    _tree: NodeSearchTree
    _tree_current: bool

    node_proximity_barrier: float

    def __init__(self, shape: List[int], sector_size: int,
                 nodes: NodeList[CortexNode], presenters: List[Presenter],
                 node_proximity_barrier: float = 0,
                 ) -> None:
        """
        Initializes a cortex with the given parameters.
        :param shape: The dimensionality of the cortex.
        :param sector_size: The shape of the sector-space.
        """
        PresenterSubject.__init__(self, presenters)
        Manager.__init__(self, CortexNode)
        self.presenters = presenters
        self.shape = shape
        self.sector_size = sector_size
        self.sector_shape = []
        self.node_proximity_barrier = node_proximity_barrier
        self.nodes = nodes
        self._tree_current = False

        for dim in shape:
            assert dim % sector_size == 0
            self.sector_shape.append(dim // sector_size)

        self.node_density_contribution = 1 # self.sector_size / (
                    # self.sector_size ** len(list(self.sector_shape)))

        self.rank = len(shape)

        self.conversion_list = [1]
        for dim in self.sector_shape[:-1]:
            self.conversion_list.append(self.conversion_list[-1] * dim)

        # Create sectors
        num_sectors = 1
        for dim in self.sector_shape:
            num_sectors *= dim
        self.sectors = []
        self.sector_densities = []

        self.sector_surroundings = []

        for sector in range(num_sectors):
            self.sectors.append([])
            self.sector_densities.append(0)
            self.sector_surroundings.append([])

        if self.nodes:
            self.refresh_tree()

    def refresh_tree(self) -> None:
        self._tree = NodeSearchTree(self.nodes)
        self._tree_current = True

    def get_random_coords(self) -> Coords:
        coords = []
        for dim in self.shape:
            coords.append(random.randint(0, dim - 1))
        return Coords(coords)

    def get_k_nearest(self, node: Union[CortexNode, Coords], k: int) -> \
            List[Tuple[float, CortexNode]]:
        return self._tree.get_k_nearest(node, k)

    def coords_to_sector(self, coords: Union[List[int], Coords]) -> Coords:
        return coords // self.sector_size

    def coords_to_sector_int(self, coords: Union[List[int], Coords]) -> int:
        sector = self.coords_to_sector(coords)
        return self.sector_to_int(sector)

    def sector_to_int(self, sector: Coords):
        s = 0
        for i in range(len(sector)):
            s += sector[i] * self.conversion_list[i]
        return int(s)

    def int_to_sector(self, val: int) -> List[int]:
        sector = []
        for i in range(len(self.conversion_list) - 1, -1, -1):
            coord = val // max((self.conversion_list[i], 1))
            sector.append(coord)
            val -= coord * max((self.conversion_list[i], 1))
        return sector[::-1]


    def _add(self, node: CortexNode):
        sector_int = self.coords_to_sector_int(node.get_coords())
        self.sector_densities[sector_int] += self.node_density_contribution
        self._tree_current = False

    def _remove(self, node: CortexNode):
        sector_int = self.coords_to_sector_int(node.get_next_coords())
        self.sector_densities[sector_int] -= self.node_density_contribution
        self._tree_current = False

        self.update('death', node)

    def move_node(self, node: CortexNode, translations: np.array):
        old_coords = node.get_coords()
        assert len(node.get_coords()) == len(translations)
        new_coords = node.get_coords() + translations

        # Proposed move is out of bounds.
        if not self.valid_coords(new_coords):
            node.move_to(node.get_coords())
            return

        # Proposed move is too close to another node.
        min_distance_to_other = self.get_k_nearest(new_coords, 1)[0][0]
        if min_distance_to_other < self.node_proximity_barrier:
            return

        node.move_to(new_coords)

        self.update('move', node)

        sector_change = self.coords_to_sector(old_coords) != self.coords_to_sector(new_coords)

        if any(sector_change):

            old_int = self.coords_to_sector_int(old_coords)
            new_int = self.coords_to_sector_int(new_coords)

            self.sector_densities[old_int] -= self.node_density_contribution
            self.sector_densities[new_int] += self.node_density_contribution

    def valid_coords(self, coords: Coords) -> bool:
        for i, coord in enumerate(coords):
            if not 0 <= coord < self.shape[i]:
                return False
        return True

    def valid_sector(self, sector: Coords) -> bool:
        for i in range(len(sector)):
            if not 0 <= sector[i] < self.sector_shape[i]:
                return False
        return True

    def get_signal_about_axial_center(
            self,
            axial_center: List[int],
            constant_dim: int,
            search_distance: int) -> float:
        search_range = range(-search_distance, search_distance + 1)
        search_incr = product(*([search_range] * (len(self.sector_shape) - 1)))
        total_signal = 0
        for var_dim_adjusts in search_incr:
            # Euclidian distance from axial centers.
            abs_dist = sum([abs(adj) ** 2 for adj in var_dim_adjusts]) \
                       ** (1 / 2)

            neighbor_sector = copy(axial_center)
            i = 0
            j = 0
            while i - j < len(var_dim_adjusts):
                adj = var_dim_adjusts[i - j]
                if i == constant_dim:
                    j += 1
                    i += 1
                    continue
                neighbor_sector[i] += adj
                i += 1

            # Ignore out-of-bounds sectors.
            if self.valid_sector(neighbor_sector):
                sector_density = (
                    self.sector_densities)[self.sector_to_int(neighbor_sector)]
            else:
                sector_density = 0

            # Scale contribution to directional signal by euclidian
            # distance from the axial center.
            # TODO REVERT
            contribution = sector_density # / (abs_dist + 1)
            total_signal += contribution
        return total_signal

    def get_surrounding_densities(self, node: CortexNode,
                                  avoid_rerun: bool = True) -> List[float]:

        core_sector_int = self.coords_to_sector_int(node.get_coords())

        # Check to see if this calculation has already been done for this
        # timestep.
        if avoid_rerun and self.sector_surroundings[core_sector_int]:
            return self.sector_surroundings[core_sector_int]
        core_sector = self.int_to_sector(core_sector_int)

        search_distance = 0
        node_found = False

        output = []
        while not node_found:
            search_distance += 1

            # Define self.rank * 2 axial centers
            axial_centers = []
            for dim in range(len(core_sector)):
                shift = np.zeros(len(core_sector))
                shift[dim] = search_distance
                axial_centers.append(core_sector + shift)
                axial_centers.append(core_sector - shift)

            # Check that at least one axial center is in bounds.
            any_inbounds = False
            for center in axial_centers:
                if self.valid_sector(center):
                    any_inbounds = True
                    break
            # If not, this is the only sector in the grid with neurons.
            if not any_inbounds:
                return [0] * len(axial_centers) + [search_distance]

            # Search regions around the axial centers.
            constant_dim_incr = 0
            output = []
            for center in axial_centers:
                total_signal = self.get_signal_about_axial_center(
                    center, round(constant_dim_incr // 2), search_distance)
                output.append(total_signal)
                constant_dim_incr += 1
            node_found = sum(output) > 0
        self.sector_surroundings[core_sector_int] = output + [search_distance]
        return output + [search_distance]
