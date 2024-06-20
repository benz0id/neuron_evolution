from abc import ABC
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Any

from src.analysis.helpers import append_csv_line, get_date_time_filename
from src.analysis.plotting import ridge_plot
from src.analysis.vis_net import draw_net
from src.entities.neuron import Neuron
from src.entities.neuron_genome import NeuronGenome
from src.entities.node import Node
from src.entities.node_list import NodeList
from src.managers.manager import Manager
from datetime import datetime
import os

from src.managers.neat_manager import NeatPopulationManager


class Neurons:
    pass


class NeuronAnalysisManager:
    """
    Manages the analysis of the state of neurons.
    """

    _get_epoch: Callable[[], int]
    _neurons: NodeList[Neuron]
    _species_to_neurons: Dict[int, List[Neuron]]
    _species_to_deaths: Dict[int, int]
    _species_to_births: Dict[int, int]

    _population_cache: Dict[str, int]
    _birth_cache: Dict[str, int]

    _get_representatives: Callable[[], Dict[int, NeuronGenome]]

    figure_out_dir: Path

    _neuron_attribute_to_getter: Dict[str, Callable[[Neuron], int]] = {
        'state': Neuron.get_state,
        'cogni': Neuron.get_cogni,
        'num_in': lambda neuron: len(neuron.get_inputs()),
        'num_out': lambda neuron: len(neuron.get_outputs()),
        'num_con': lambda neuron: len(neuron.get_outputs()) +
                                  len(neuron.get_inputs()),
        'generosity': Neuron.get_generosity,
        'spawn_investment': Neuron.get_spawn_investment,
        'connection_willingness': Neuron.get_connection_willingness,
        'received_rewards': lambda neuron: sum([reward.cogni
                                                for reward in
                                                neuron.get_rewards()]),
        'maintenance': Neuron.get_maintenance_value
    }

    def _get_num_births(self, pop_name: str, pop: List[Neuron]) -> int:
        count = 0
        for neuron in pop:
            if neuron.get_creation_epoch() == self._get_epoch():
                count += 1
        self._birth_cache[pop_name] = count
        return count

    def _get_last_pop_size(self, pop_name: str):
        if pop_name in self._population_cache:
            return self._population_cache[pop_name]
        else:
            self._population_cache[pop_name] = 0
            return 0

    def _get_num_deaths(self, pop_name: str, pop: List[Neuron]):
        # Population cache is from last epoch, birth cache is current.
        return self._get_last_pop_size(pop_name) - len(pop) \
            + self._birth_cache[pop_name]

    def _get_pop_size(self, pop_name: str, pop: List[Neuron]):
        self._population_cache[pop_name] = len(pop)
        return len(pop)

    # Do not rearrange these. Order matters due to caching.
    _population_attribute_to_getter: \
        List[Tuple[str, Callable[[Any, str, List[Neuron]], int]]] = \
        [
            ('births', _get_num_births),
            ('deaths', _get_num_deaths),
            ('population_size', _get_pop_size)
        ]

    def __init__(self,
                 data_out_dir: Path,
                 figure_out_dir: Path,
                 get_epoch: Callable[[], int],
                 neurons: NodeList[Neuron],
                 genome_manager: NeatPopulationManager):

        self._get_epoch = get_epoch
        self._neurons = neurons
        self._species_to_neurons = genome_manager.get_species_dict()
        self._species_to_deaths = {}
        self._species_to_births = {}
        self._get_representatives = genome_manager.get_representatives
        self.genome_manager = genome_manager

        self._population_cache = {}
        self._birth_cache = {}
        formatted_date_time = get_date_time_filename()
        self.data_out_dir = data_out_dir / formatted_date_time
        self.figure_out_dir = figure_out_dir / formatted_date_time

    def _get_pops(self) -> List[Tuple[str, List[Neuron]]]:
        pops = []
        pops.append(('all_neurons', self._neurons))
        for species in self._species_to_neurons:
            pops.append((f'species_{species}',
                         self._species_to_neurons[species]))
        return pops

    def _get_header(self) -> List[str]:
        header = ['epoch']
        for attr, method in self._neuron_attribute_to_getter.items():
            header.append(attr)

        for attr, method in self._population_attribute_to_getter:
            header.append(attr)
        return header

    def store_simulation_stats(self) -> None:
        if not self.data_out_dir.exists():
            os.mkdir(self.data_out_dir)

        pops = self._get_pops()

        for pop_name, pop in pops:
            to_write = []
            to_write.append(self._get_epoch())
            # Add average of each attribute to csv file.
            for attr, method in self._neuron_attribute_to_getter.items():
                total = 0
                for neuron in pop:
                    total += method(neuron)
                to_write.append(total / (len(pop) + 1))

            for attr, method in self._population_attribute_to_getter:
                to_write.append(method(self, pop_name, pop))

            outfile = self.data_out_dir / (pop_name + '.csv')

            if not outfile.exists():
                append_csv_line(outfile, self._get_header())
            append_csv_line(outfile, to_write)

    def generate_species_figures(self):
        genomes = self.genome_manager.get_representatives()
        fig_out = self.figure_out_dir / 'representatives'
        if not fig_out.exists():
            os.mkdir(fig_out)
        for species, genome in genomes.items():
            outfile = fig_out / f'{species}.svg'
            draw_net(self.genome_manager.config, genome, filename=outfile)

    def compile_figures(self):
        if not self.figure_out_dir.exists():
            os.mkdir(self.figure_out_dir)

        self.generate_species_figures()

        for pop_name, pop in self._get_pops():
            fig_out = self.figure_out_dir / 'attributes'
            if not fig_out.exists():
                os.mkdir(fig_out)

            outfile = self.data_out_dir / (pop_name + '.csv')
            figure_out = fig_out / (pop_name + '.jpg')
            ridge_plot(outfile, figure_out)


