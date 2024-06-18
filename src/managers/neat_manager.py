from __future__ import print_function

from itertools import count
from typing import List, Dict

from src.entities.neuron import Neuron
from src.entities.neuron_genome import NeuronGenome
from src.entities.node_list import NodeList
from src.managers.manager import Manager
from neat.reporting import ReporterSet


class NeatPopulationManager(Manager):
    """
    Adapts required functionality from the neat-python module.
    Allows for the maintenance of a continuous population of genomes.
    """

    _species_current: bool

    _neurons: NodeList[Neuron]
    _species_to_neurons: Dict[int, List[Neuron]]

    def __init__(self, config,
                 neurons: NodeList[Neuron],
                 initial_state=None,):
        super().__init__(Neuron)
        self._neurons = neurons
        self.reporters = ReporterSet()
        self.config = config
        self.ancestors = {}
        self.genome_indexer = count(1)
        self.generation = 0
        self._species_to_neurons = {}

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.species = config.species_set_type(config.species_set_config,
                                                   self.reporters)
            self.population = {}
        else:
            self.population, self.species, self.generation = initial_state

        self._species_current = False

    def _remove(self, neuron: Neuron) -> None:
        self.kill([neuron.get_genome_id()])
        self._species_to_neurons[neuron.get_species_id()].remove(neuron)

    def _add(self, neuron: Neuron) -> None:
        pass

    def species_current(self):
        return self._species_current

    def new_genome(self) -> NeuronGenome:

        key = next(self.genome_indexer)
        g = self.config.genome_type(key)
        g.configure_new(self.config.genome_config)
        g.id = key
        self.ancestors[key] = tuple()
        self.population[key] = g

        self._species_current = False

        return g

    def create_new_population(self, num_genomes) -> None:
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = self.config.genome_type(key)
            g.configure_new(self.config.genome_config)
            g.id = key
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        self.population = new_genomes

        self._species_current = False

    def kill(self, genome_ids: List[int]):
        for gid in genome_ids:
            del self.population[gid]
            species = self.species.genome_to_species[gid]
            del self.species.genome_to_species[gid]
            del self.species.species[species].members[gid]

            if len(self.species.species[species].members) == 0:
                del self.species.species[species]

        self._species_current = False

    def reproduce_sexually(self, p1: Neuron, p2: Neuron) -> NeuronGenome:
        p1_id = p1.get_genome_id()
        p2_id = p2.get_genome_id()

        p1_species = p1.get_species_id()
        p2_species = p2.get_species_id()
        if not p1_species == p2_species:
            raise ValueError(
                f"Parent 1 and Parent 2 Belong to different species."
                f"{p1_id} <3 {p2_id} but {p1_species} != {p2_species}")

        p1 = self.population[p1_id]
        p2 = self.population[p2_id]

        gid = next(self.genome_indexer)
        child = self.config.genome_type(gid)
        child.configure_crossover(p1, p2, self.config.genome_config)
        child.mutate(self.config.genome_config)
        self.population[gid] = child
        self.ancestors[gid] = (p1.get_genome_id(),  p2.get_genome_id())
        child.id = gid

        self._species_current = False

        return child

    def reproduce_asexually(self, neuron: Neuron) -> NeuronGenome:
        return self.reproduce_sexually(neuron, neuron)

    def classify_species(self):
        self.species.speciate(self.config, self.population, self.generation)
        self.generation += 1
        for s in self.species.species:
            for m in self.species.species[s].members:
                self.species.species[s].members[m].set_species(s)
        self._species_current = True

        def add(d, k, v):
            if k in d:
                d[k].append(v)
            else:
                d[k] = [v]

        for species in self._species_to_neurons:
            self._species_to_neurons[species] = []
        for neuron in self._neurons:
            add(self._species_to_neurons, neuron.get_species_id(), neuron)

    def get_species_ids(self) -> List[int]:
        return list(self._species_to_neurons.keys())

    def get_population(self, species: int) -> List[Neuron]:
        return self._species_to_neurons[species]

    def get_species_dict(self) -> Dict[int, List[Neuron]]:
        return self._species_to_neurons

