from typing import List, Callable, Dict

import numpy as np
from src.entities.neuron import Neuron, Coords
from src.entities.neuron_genome import NeuronGenome
from src.entities.node import Node
from src.entities.node_list import NodeList
from src.managers.cortex_manager import CortexManager, NodeSearchTree
from src.managers.manager import Manager
from src.managers.neat_manager import NeatPopulationManager

from src.presenters.matrix_presenter import PresenterSubject, Presenter


def add(dic, key, val):
    if key in dic:
        dic[key].append(val)
    else:
        dic[key] = [val]


class NeuronCreationManager(PresenterSubject, Manager):
    """
    Responsible for the creation and reproduction of neurons.
    """

    partner_search_radius: float
    get_epoch: Callable[[], int]
    get_node_id: Callable[[], int]
    neat: NeatPopulationManager
    spawn_radius: int
    cortex: CortexManager
    starting_cogni: float
    add_neuron: Callable[[Neuron], None]
    neurons: NodeList[Neuron]
    breeding_threshold: float

    target_population_size: int
    max_to_add: int
    min_population_size: int
    starting_population_size: int

    def __init__(self, partner_search_radius: float,
                 neat_manager: NeatPopulationManager,
                 get_epoch: Callable[[], int],
                 get_node_id: Callable[[], int],
                 add_neuron: Callable[[Neuron], None],
                 spawn_radius: int,
                 cortex: CortexManager,
                 starting_cogni: float,
                 neurons: NodeList[Neuron],
                 presenters: List[Presenter],
                 breeding_threshold: float,
                 starting_population_size: int,
                 target_population_size: int = 0,
                 max_to_add: int = 0,
                 minimum_population_size: int = 0) -> None:

        PresenterSubject.__init__(self, presenters)
        Manager.__init__(self, Node)
        self.starting_population_size = starting_population_size
        self.target_population_size = target_population_size
        self.min_population_size = minimum_population_size
        self.max_to_add = max_to_add
        self.neurons = neurons
        self.partner_search_radius = partner_search_radius
        self.get_epoch = get_epoch
        self.neat = neat_manager
        self.spawn_radius = spawn_radius
        self.cortex = cortex
        self.starting_cogni = starting_cogni
        self.get_node_id = get_node_id
        self.add_neuron = add_neuron
        self.breeding_threshold = breeding_threshold

    def spawn_neuron(self, coords: Coords = None, random_shift: bool = False) -> Neuron:
        genome = self.neat.new_genome()
        cogni = self.starting_cogni
        if coords is None:
            coords = self.cortex.get_random_coords()
        return self.build_child(genome, cogni, coords, random_shift)

    def spawn_neurons(self, n: int):
        return [self.spawn_neuron() for _ in range(n)]

    def breed_willing_neurons(self, neurons: NodeList[Neuron]) -> List[Neuron]:
        """
        Breeds the willing neurons.
        :param neurons:
        :return:
        """
        willing_neurons = []
        for neuron in neurons:
            # Willing to invest more than the minimum?
            if neuron.get_spawn_investment() > self.breeding_threshold:
                willing_neurons.append(neuron)
                neuron.set_willingness(True)
        return self.breed_neurons(willing_neurons)

    def breed_neurons(self, neurons: List[Neuron]) -> List[Neuron]:
        """
        :param neurons: Maps a neuron to its contribution to the child.
        :return:
        """
        children = []

        species_to_neurons = {}
        for neuron in neurons:
            add(species_to_neurons, neuron.get_species_id(), neuron)

        for species in species_to_neurons:
            population = species_to_neurons[species]
            couples, singletons = self.make_couples(population)

            for p1, p2 in couples:
                children.append(self.breed(p1, p2))

            for p in singletons:
                children.append(self.clone(p))

        return children

    def breed(self, p1: Neuron, p2: Neuron) -> Neuron:
        cogni = p1.procreate()
        cogni += p2.procreate()

        assert p1.get_creation_epoch() != self.get_epoch()
        assert p2.get_creation_epoch() != self.get_epoch()

        self.update('mate', p1)
        self.update('mate', p2)

        genome = self.neat.reproduce_sexually(p1, p2)

        coords = (p1.get_coords() + p2.get_coords()) // 2

        child = self.build_child(genome, cogni, coords)

        return child

    def clone(self, parent: Neuron):

        assert parent.get_creation_epoch() != self.get_epoch()

        cogni = parent.procreate()
        self.update('clone', parent)

        genome = self.neat.reproduce_asexually(parent)

        coords = parent.get_coords()

        child = self.build_child(genome, cogni, coords)

        return child

    def build_child(self, genome: NeuronGenome, cogni: float,
                    coord_center: Coords, random_shift: bool = True) -> Neuron:

        if random_shift:
            new_coords = Coords([-1] * len(coord_center))
            while not self.cortex.valid_coords(new_coords):
                random_translation = np.random.randint(
                                        -self.spawn_radius,
                                        self.spawn_radius,
                                        size=(len(coord_center)))

                new_coords = coord_center + random_translation
        else:
            new_coords = coord_center

        child = Neuron(
            genome, new_coords, self.get_epoch(), cogni, self.get_node_id()
        )

        self.update('born', child)
        self.add_neuron(child)
        return child

    def maintain_population(self):
        """
        If the current neuron population is less than the target population,
        add new neurons with random genomes.
        :param target_size: The target population size.
        :param max_new: Max number of neurons to add to reach the target size.
        :return: None
        """
        if len(self.neurons) < self.target_population_size:
            to_add = min([self.max_to_add, self.target_population_size -
                          len(self.neurons)])
            for _ in range(to_add):
                self.spawn_neuron()

        while len(self.neurons) < self.min_population_size:
            self.spawn_neuron()

    def make_couples(self, neurons: List[Neuron]) -> \
            tuple[list[tuple[Neuron, Neuron]], list[Neuron]]:
        tree = NodeSearchTree(neurons)

        couples = []
        singles = []

        i = 0
        while i < len(neurons):
            neuron = neurons[i]

            assert neuron.get_creation_epoch() != self.get_epoch()

            if not neuron.is_willing():
                i += 1
                continue

            partner_found = False
            best_partners = tree.get_in_radius(neuron,
                                               self.partner_search_radius)
            for dist, partner in best_partners:
                if partner is neuron:
                    continue

                if partner.is_willing():
                    couples.append((neuron, partner))
                    neuron.set_willingness(False)
                    partner.set_willingness(False)
                    partner_found = True

            if not partner_found:
                singles.append(neuron)
                neuron.set_willingness(False)

        return couples, singles

    def spawn_initial_population(self) -> None:
        while len(self.neurons) < self.starting_population_size:
            self.spawn_neuron()

    def _remove(self, node: Node) -> None:
        pass

    def _add(self, neuron: Node) -> None:
        pass








