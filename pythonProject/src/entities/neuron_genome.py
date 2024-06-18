from __future__ import print_function

from typing import Callable

from neat.genome import DefaultGenome
from neat.six_util import iteritems

# TODO
standin_max_radius = 10


class NeuronGenome(DefaultGenome):

    species: int = -1
    id: int

    get_fitness: Callable[[], float] = None

    def set_species(self, species: int):
        self.species = species

    def get_species(self) -> int:
        return self.species

    def get_max_radius(self) -> float:
        return standin_max_radius

    def get_genome_id(self) -> int:
        return self.id

    def set_fitness_method(self, fitness_function: Callable[[], float]):
        self.get_fitness = fitness_function

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes.

        Virtually the same as the existing method, but abstracting the fitness
        to another module.

        """
        g1_fitness = genome1.get_fitness()
        g2_fitness = genome2.get_fitness()

        assert isinstance(g1_fitness, (int, float))
        assert isinstance(g2_fitness, (int, float))
        if g1_fitness > g2_fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)
