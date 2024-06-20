from multiprocessing import Pool
from pathlib import Path

import neat
import src

from src.analysis.analysis_manager import NeuronAnalysisManager
from src.analysis.timer import PipelineTimer
from src.managers.epoch_counter import EpochCounter
from src.entities.neuron import CortexNode, Neuron
from src.entities.neuron_genome import NeuronGenome
from src.managers.cofiring_manager import CofiringManager
from src.managers.cogni_cost import CogniCostManager
from src.managers.connection_manager import ConnectionManager
from src.managers.cortex_manager import CortexManager
from src.managers.helpers.manager_set import ManagerSet
from src.managers.neat_manager import NeatPopulationManager
from src.managers.node_manager import NodeManager

from src.managers.creation_manager import NeuronCreationManager
from src.presenters.matrix_presenter import MatrixPresenter

# Breeding Params.
SPAWN_RADIUS = 10
PARTNER_SEARCH_RADIUS = 16
BREEDING_THRESHOLD = 450
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

# Firing params.
NEURON_FIRING_MEMORY = 50

# Costs
RELATIONSHIP_COST = 0
EXISTENCE_COST = 1
STATE_COST = 10
MOVEMENT_COST = 1

# Display Params
MATRIX_DISPLAY = False
WIDTH = 64
HEIGHT = 32
UP_EVERY = 10
FPS = 20
PALETTE = 255
NUM_RESERVED = 10
VERBOSE = False
PORT = '/dev/tty.usbmodem101'

# Simulation meta params
SIMULATION_OUT_DIR = Path('simulation/data/sim_stats')
FIGURES_OUT_DIR = Path('simulation/data/figures')
TIMING_OUT_DIR = Path('simulation/data/timing')
MAX_NUM_EPOCHS = 10000
SPECIATE_EVERY = 100


# Population Params
MIN_POPULATION = 10000
TARGET_POPULATION = 10000
MAX_TO_ADD = 10000
STARTING_POPULATION_SIZE = 10000

# Timing args.
DO_TIMING = True
PRINT_TIMES = True

# Other
NUM_CORES = 8

# Config
config_path = Path('config')
config = neat.Config(NeuronGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
src.managers.neat_manager.CONFIG = config

def get_managers() -> ManagerSet:

    nodes = NodeManager()

    pool = Pool(NUM_CORES)

    pres = []
    if MATRIX_DISPLAY:
        presenter = MatrixPresenter(
            WIDTH,
            HEIGHT,
            UP_EVERY,
            FPS,
            PALETTE,
            NUM_RESERVED,
            VERBOSE,
            PORT
        )
        pres.append(presenter)

    epoch_counter = EpochCounter(pres, MAX_NUM_EPOCHS)
    genome_manager = NeatPopulationManager(nodes.get_all(Neuron),
                                           epoch_counter.get_epoch,
                                           pool,
                                           speciate_every=SPECIATE_EVERY,)
    nodes.add_manager(genome_manager)

    action_timer = PipelineTimer(TIMING_OUT_DIR, 'action',
                                 epoch_counter.get_epoch, PRINT_TIMES, DO_TIMING)
    update_timer = PipelineTimer(TIMING_OUT_DIR, 'update',
                                 epoch_counter.get_epoch, PRINT_TIMES, DO_TIMING)

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
        epoch_counter.get_epoch,
        nodes.get_next_id,
        nodes.add_node,
        SPAWN_RADIUS,
        cortex,
        BASE_COGNI,
        nodes.get_all(Neuron),
        pres,
        breeding_threshold=BREEDING_THRESHOLD,
        starting_population_size=STARTING_POPULATION_SIZE,
        target_population_size=TARGET_POPULATION,
        max_to_add=MAX_TO_ADD,
        minimum_population_size=MIN_POPULATION
    )

    connections = ConnectionManager(
        CONNECTION_DECAY_RATE,
        CONNECTION_START_VALUE,
        COGNI_TO_CON_CONVERSION,
        DIVIDEND_SCALING_FACTOR,
        cortex,
        WILLINGNESS_THRESHOLD,
        pres,
        nodes.get_all(Neuron)
    )

    firing = CofiringManager(
        NEURON_FIRING_MEMORY,
        epoch_counter.get_epoch,
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

    analysis = NeuronAnalysisManager(
        SIMULATION_OUT_DIR,
        FIGURES_OUT_DIR,
        epoch_counter.get_epoch,
        nodes.get_all(Neuron),
        genome_manager
    )

    return ManagerSet(
        nodes=nodes,
        epoch_counter=epoch_counter,
        genome_manager=genome_manager,
        cortex=cortex,
        creator=creator,
        connections=connections,
        firing=firing,
        costs=costs,
        analysis=analysis,
        action_timer=action_timer,
        update_timer=update_timer
    )
