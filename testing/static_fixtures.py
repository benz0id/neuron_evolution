from pathlib import Path

import neat
from entities.counter import EpochCounter
from entities.neuron import CortexNode, Neuron
from entities.neuron_genome import NeuronGenome
from managers.cofiring_manager import CofiringManager
from managers.cogni_cost import CogniCostManager
from managers.connection_manager import ConnectionManager
from managers.cortex_manager import CortexManager
from managers.helpers.manager_set import ManagerSet
from managers.neat_manager import NeatPopulationManager
from managers.node_manager import NodeManager

from src.managers.creation_manager import NeuronCreationManager
from src.presenters.matrix_presenter import MatrixPresenter

# Breeding Params.
SPAWN_RADIUS = 10
PARTNER_SEARCH_RADIUS = 16
BREEDING_THRESHOLD = 75
BASE_COGNI = 100

# Cortex params.
DIMS = [64, 32]
SECTOR_SIZE = 8

# Relationship maintenance
CONNECTION_DECAY_RATE = 1
CONNECTION_START_VALUE = 50
COGNI_TO_CON_CONVERSION = 1
DIVIDEND_SCALING_FACTOR = 2

# Relationship formation
WILLINGNESS_THRESHOLD = 7

NEURON_FIRING_MEMORY = 50

# Costs
RELATIONSHIP_COST = 0
EXISTENCE_COST = 1
STATE_COST = 10
MOVEMENT_COST = 1

# Display Params
WIDTH = 64
HEIGHT = 32
UP_EVERY = 10
FPS = 20
PALETTE = 255
NUM_RESERVED = 9
VERBOSE = False



def get_managers(matrix_presenter: bool = True) -> ManagerSet:
    config_path = Path('../config')

    config = neat.Config(NeuronGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    nodes = NodeManager()

    pres = []
    if matrix_presenter:
        pres.append(MatrixPresenter(
            WIDTH,
            HEIGHT,
            nodes.get_all(CortexNode),
            UP_EVERY,
            FPS,
            PALETTE,
            NUM_RESERVED,
            VERBOSE,
            '/dev/tty.usbmodem1101'
        ))

    epoch_counter = EpochCounter(pres)
    genome_manager = NeatPopulationManager(config)
    nodes.add_manager(genome_manager)

    cortex = CortexManager(
        DIMS,
        SECTOR_SIZE,
        nodes.get_all(CortexNode),
        pres
    )
    nodes.add_manager(cortex)

    creator = NeuronCreationManager(
        PARTNER_SEARCH_RADIUS,
        genome_manager,
        epoch_counter._get_epoch,
        nodes.get_next_id,
        nodes.add_node,
        SPAWN_RADIUS,
        cortex,
        BASE_COGNI,
        nodes.get_all(Neuron),
        pres,
        BREEDING_THRESHOLD
    )

    connections = ConnectionManager(
        CONNECTION_DECAY_RATE,
        CONNECTION_START_VALUE,
        COGNI_TO_CON_CONVERSION,
        DIVIDEND_SCALING_FACTOR,
        cortex,
        WILLINGNESS_THRESHOLD,
        pres
    )

    firing = CofiringManager(
        NEURON_FIRING_MEMORY,
        epoch_counter._get_epoch,
        pres
    )

    costs = CogniCostManager(
        nodes.get_all(Neuron),
        nodes.remove_node,
        RELATIONSHIP_COST,
        EXISTENCE_COST,
        STATE_COST,
        MOVEMENT_COST
    )

    return ManagerSet(
        nodes=nodes,
        epoch_counter=epoch_counter,
        genome_manager=genome_manager,
        cortex=cortex,
        creator=creator,
        connections=connections,
        firing=firing,
        costs=costs
    )
