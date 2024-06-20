from __future__ import print_function

import sys
from itertools import count
from multiprocessing import Pool
from typing import List, Dict, Callable

import numpy as np

from src.analysis.timer import PipelineTimer
from src.entities.neuron import Neuron, Actions
from src.entities.neuron_genome import NeuronGenome
from src.entities.node_list import NodeList
from src.managers.manager import Manager
from neat.reporting import ReporterSet
from neat.nn import FeedForwardNetwork
from neat.species import Species

CONFIG = None
def add(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def get_output_vec(input_vec: np.array,
                   genome: NeuronGenome) -> np.array:
    net = FeedForwardNetwork.create(genome, CONFIG)
    if len(input_vec) != 14:
        pass
    res = net.activate(input_vec)
    return res


class NeatPopulationManager(Manager):
    """
    Adapts required functionality from the neat-python module.
    Allows for the maintenance of a continuous population of genomes.
    """

    _species_current: bool
    _config: None

    _neurons: NodeList[Neuron]
    _species_to_neurons: Dict[int, List[Neuron]]
    speciate_every: int
    get_epoch: Callable[[], int]
    _pool: Pool


    def __init__(self,
                 neurons: NodeList[Neuron],
                 get_epoch: Callable[[], int],
                 pool: Pool,
                 initial_state=None,
                 speciate_every: int = 1):
        super().__init__(Neuron)
        self._pool = pool
        self.speciate_every = speciate_every
        self.get_epoch = get_epoch
        self._neurons = neurons
        self.reporters = ReporterSet()
        self.config = CONFIG
        self.ancestors = {}
        self.genome_indexer = count(1)
        self.generation = 0
        self._species_to_neurons = {}

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.species = self.config.species_set_type(self.config.species_set_config,
                                                   self.reporters)
            self.population = {}
        else:
            self.population, self.species, self.generation = initial_state

        self._species_current = False

    def _remove(self, neuron: Neuron) -> None:
        self.kill([neuron.get_genome_id()])
        # Check that neuron has been assigned to a species.
        if neuron.get_species_id() != -1:
            self._species_to_neurons[neuron.get_species_id()].remove(neuron)
            if not self._species_to_neurons[neuron.get_species_id()]:
                del self._species_to_neurons[neuron.get_species_id()]

    @classmethod
    def set_config(cls, config):
        cls.config = config

    def _add(self, neuron: Neuron) -> None:
        pass

    def species_current(self):
        return self._species_current


    def get_input_vec(self, neuron: Neuron):
        cogni = neuron.get_cogni()
        num_in = len(neuron.get_inputs())
        num_out = len(neuron.get_outputs())
        state = neuron.get_state()
        hidden = neuron.get_hidden()
        densities = neuron.get_surroundings()
        distance = neuron.get_neuron_dist()

        # TODO temporary. allow neurons to perform logic on inputs.
        if neuron.get_inputs():
            mean_input = sum([inp.get_state() for inp in neuron.get_inputs()]) \
                         / len(neuron.get_inputs())
        else:
            mean_input = 0

        input_vec = np.array([
            cogni,
            num_in,
            num_out,
            state,
            *hidden,
            *densities,
            distance,
            mean_input
        ])

        return input_vec

    def convert_to_actions(self, output_vec: np.array):
        translation = output_vec[0:2]
        next_state = output_vec[2]
        next_hidden_state = output_vec[3:7]
        generosity = output_vec[7]
        maintainance_val = output_vec[8]
        conn_willingness = output_vec[9]
        spawn_investment = output_vec[10]

        return Actions(
            translation=translation,
            generosity=generosity,
            maintenance_value=maintainance_val,
            connection_willingness=conn_willingness,
            next_state=next_state,
            next_hidden_state=next_hidden_state,
            spawn_investment=spawn_investment,
            learning_rate=0
        )

    def update_all_actions(self, timer: PipelineTimer):

        timer.begin('get_inputs')
        input_vecs = [self.get_input_vec(neuron)
                      for neuron in self._neurons]

        timer.begin('get_genomes')
        genomes = [self.population[neuron.get_genome_id()]
                   for neuron in self._neurons]

        """
        timer.begin('get_nets')
        nets = [FeedForwardNetwork.create(genome, self.config)
                for genome in genomes]"""

        timer.begin('compute_output')
        args = zip(input_vecs, genomes)

        output_vecs = self._pool.starmap(get_output_vec, args)
        # output_vecs = [get_output_vec(*arg) for arg in args]

        timer.begin('convert_to_actions')
        actions = [self.convert_to_actions(output_vec)
                   for output_vec in output_vecs]

        for i, neuron in enumerate(self._neurons):
            neuron.set_actions(actions[i])
        timer.end()

    def new_genome(self) -> NeuronGenome:

        key = next(self.genome_indexer)
        g = self.config.genome_type(key)
        g.configure_new(self.config.genome_config)
        g.id = key
        self.ancestors[key] = tuple()
        self.population[key] = g

        self._species_current = False

        return g

    def force_assign_next_species(self, neuron: Neuron) -> None:
        g = self.population[neuron.get_genome_id()]
        self._species_current = False

        # Assign the new genome to a new species.
        if self._species_to_neurons.keys():
            max_species_id = max(self._species_to_neurons.keys()) + 1
        else:
            max_species_id = 0
        g.species = max_species_id
        self.species.genome_to_species[g.id] = g.species

        self.species.species[max_species_id] = \
            Species(max_species_id, self.generation)
        self.species.species[max_species_id].members[neuron.get_genome_id()] = g
        self.species.species[max_species_id].representative = g

        add(self._species_to_neurons, neuron.get_species_id(), neuron)


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
            if gid not in self.species.genome_to_species:
                print('Attempting to delete neuron without speciated genome.',
                      file=sys.stderr)
                continue
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

        self.species.species[p1_species].members[gid] = child
        self.species.genome_to_species[child.id] = p1_species

        add(self._species_to_neurons, child.get_species_id(), child)

        self._species_current = False

        return child

    def reproduce_asexually(self, neuron: Neuron) -> NeuronGenome:
        return self.reproduce_sexually(neuron, neuron)

    def classify_species(self):

        return

        # Only speciate after a predef number of epochs.
        if self.get_epoch() % self.speciate_every != 0:
            return

        # Only speciate if we need to.
        if self._species_current:
            return

        print('Beginning Speciation')
        self.species.speciate(self.config, self.population, self.generation)
        self.generation += 1
        for s in self.species.species:
            for m in self.species.species[s].members:
                self.species.species[s].members[m].set_species(s)
        self._species_current = True

        for species in self._species_to_neurons:
            self._species_to_neurons[species] = []
        for neuron in self._neurons:
            add(self._species_to_neurons, neuron.get_species_id(), neuron)
        print(f'Speciation Complete: {len(self._species_to_neurons)} Species Now Exist')

    def get_representatives(self):
        representatives = {}
        for sid, species in self.species.species.items():
            representatives[sid] = species.representative
        return representatives

    def get_species_ids(self) -> List[int]:
        return list(self._species_to_neurons.keys())

    def get_population(self, species: int) -> List[Neuron]:
        return self._species_to_neurons[species]

    def get_species_dict(self) -> Dict[int, List[Neuron]]:
        return self._species_to_neurons

