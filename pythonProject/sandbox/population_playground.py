from pathlib import Path

import neat
from entities.neuron_genome import NeuronGenome
from managers.neat_manager import NeatPopulationManager

config_path = Path('../config')

config = neat.Config(NeuronGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


def dummy_fitness_method(x):
    return 5


manager = NeatPopulationManager(config, dummy_fitness_method)
manager.create_new_population(10)
manager.classify_species()

# manager.reproduce_sexually([(3, 9)])