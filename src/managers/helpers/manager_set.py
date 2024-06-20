from src.analysis.analysis_manager import NeuronAnalysisManager
from src.analysis.timer import PipelineTimer
from src.managers.epoch_counter import EpochCounter
from src.managers.cofiring_manager import CofiringManager
from src.managers.cogni_cost import CogniCostManager
from src.managers.connection_manager import ConnectionManager
from src.managers.cortex_manager import CortexManager
from src.managers.creation_manager import NeuronCreationManager
from src.managers.neat_manager import NeatPopulationManager
from src.managers.node_manager import NodeManager


class ManagerSet:
    nodes: NodeManager
    epoch_counter: EpochCounter
    genomes: NeatPopulationManager
    cortex: CortexManager
    creator: NeuronCreationManager
    connections: ConnectionManager
    firing: CofiringManager
    costs: CogniCostManager
    analysis: NeuronAnalysisManager

    def __init__(self, nodes: NodeManager, epoch_counter: EpochCounter,
                 genome_manager: NeatPopulationManager, cortex: CortexManager,
                 creator: NeuronCreationManager, connections: ConnectionManager,
                 firing: CofiringManager, costs: CogniCostManager,
                 analysis: NeuronAnalysisManager, action_timer: PipelineTimer,
                 update_timer: PipelineTimer):
        self.nodes = nodes
        self.epoch_counter = epoch_counter
        self.genomes = genome_manager
        self.cortex = cortex
        self.creator = creator
        self.connections = connections
        self.firing = firing
        self.costs = costs
        self.analysis = analysis
        self.action_timer = action_timer
        self.update_timer = update_timer
