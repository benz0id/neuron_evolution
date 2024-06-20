from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Union

import numpy as np

from src.entities.cortex_node import CortexNode
from src.entities.neuron_genome import NeuronGenome
from src.entities.node import Node, Coords

#TODO
FIRING_THRESHOLD = 5

def get_distance(c1: Coords, c2: Coords) -> float:
    return np.sum((c1 - c2) ** 2) ** (1 / 2)

@dataclass
class Reward:
    sender: Node
    cogni: float
    distance: int
    distributed: bool = field(default=False)


@dataclass
class CofireRecord:
    creation_epoch: int

    # All input neurons that fired in this state.
    input_states: List[Tuple[Node, float]]

    # All output neurons that may have fired as a result.
    output_found: bool = field(default=False)
    resultant_output_states: Dict[Node, float] = field(default_factory=dict)

    # Rewards that have been attributed to decisions made during this cofire.
    # Bool indicates whether the reward has been propogated.
    rewards: List[Reward] = field(default_factory=list)

    age: int = field(default=0)

    def __hash__(self):
        return self.creation_epoch


@dataclass
class NodeStateAttrs:

    # The states of the neuron.
    state: float
    hidden_state: np.array

    # The cogni available to the neuron.
    cogni: float

    # Coordinates of the neuron.
    coords: Coords

    # Rewards received during this cycle.
    rewards: List[Reward] = field(default_factory=list)

    # Inputs that lead the neuron to fire this round.
    last_input_states: Union[CofireRecord, None] = field(default=None)

    # Input neurons mapped to their relationship level.
    inputs: Dict[Node, float] = field(default_factory=dict)

    # Output neurons.
    outputs: List[Node] = field(default_factory=list)


@dataclass
class Actions:
    """
    Stores the action descisions of the neuron at each timestep.
    """

    translation: np.array

    # How much to reward an input neuron for a rewarding cofiring.
    generosity: float

    # How much cogni to invest in all connections.
    maintenance_value: int

    # How many inputs should be added or removed.
    connection_willingness: int

    # The state of the neuron in the next iteration.
    next_state: float
    next_hidden_state: np.array

    # How much to invest in a child.
    spawn_investment: int

    # NN Hypers
    learning_rate: float


class Neuron(CortexNode):

    _creation_epoch: int
    _genome: NeuronGenome

    _attrs: NodeStateAttrs

    # Copied from attrs at each epoch, replaced at the end of the epoch.
    _next_attrs: NodeStateAttrs

    _actions: Actions

    _cofiring_records: deque[CofireRecord]
    _to_distribute: Set[CofireRecord]

    _willing: bool

    _candidate_record: Union[CofireRecord, None]

    surroundings: np.array
    neuron_dist: int

    def __init__(self, genome: NeuronGenome, starting_coords: Coords,
                 creation_epoch: int, cogni: float, node_id: int,
                 hidden_state_len: int = 4):
        super().__init__(creation_epoch, node_id)

        self.surroundings = []
        self.neuron_dist = -1

        self._genome = genome
        self._genome.set_fitness_method(self.get_cogni)

        self._attrs = NodeStateAttrs(
            state=0,
            hidden_state=np.zeros(hidden_state_len),
            cogni=cogni,
            coords=starting_coords
        )
        self._next_attrs = deepcopy(self._attrs)
        self._actions = Actions(0, 0, 0, 0, 0, np.zeros(hidden_state_len), 0, 0)

        self._willing = False

        self._cofiring_records = deque()
        self._to_distribute = set()

    def get_translation(self) -> np.array:
        return self._actions.translation

    def get_hidden(self) -> np.array:
        return self._attrs.hidden_state

    def set_hidden(self, hidden: np.array):
        self._next_attrs.hidden_state = hidden

    def set_state(self, state: float):
        self._next_attrs.state = state

    def get_genome_id(self) -> int:
        return self._genome.id

    def set_actions(self, actions: Actions):
        self._actions = actions

    def get_connection_willingness(self) -> float:
        return self._actions.connection_willingness

    def get_rewards(self) -> List[Reward]:
        return self._attrs.rewards

    def add_candidiate_record(self, record: CofireRecord):
        self._candidate_record = record

    def get_generosity(self) -> float:
        return self._actions.generosity

    def get_candidiate_record(self):
        return self._candidate_record

    def del_candidiate_record(self):
        self._candidate_record = None

    def add_cofire_record(self, record: CofireRecord):
        self._cofiring_records.append(record)

    def get_cofire_records(self) -> deque[CofireRecord]:
        return self._cofiring_records

    def get_species_id(self) -> int:
        return self._genome.get_species()

    def get_cogni(self) -> float:
        return self._attrs.cogni

    def get_next_cogni(self) -> float:
        return self._attrs.cogni

    def get_last_input_state(self) -> CofireRecord:
        return self._attrs.last_input_states

    def set_input_state_record(self, record: Union[CofireRecord, None]):
        self._next_attrs.last_input_states = record

    def get_max_radius(self) -> float:
        return self._genome.get_max_radius()

    def is_willing(self):
        return self._willing

    def set_willingness(self, willingness: bool):
        self._willing = willingness

    def procreate(self) -> float:
        self.sub_cogni(self._actions.spawn_investment)
        return self._actions.spawn_investment

    def get_spawn_investment(self) -> float:
        return self._actions.spawn_investment

    def add_output(self, output: Node):
        if output not in self._next_attrs.outputs:
            self._next_attrs.outputs.append(output)
            output.add_input(self)

    def remove_output(self, output: Node):
        if output in self._next_attrs.outputs:
            self._next_attrs.outputs.remove(output)
            output.remove_input(self)

    def get_outputs(self) -> List[Node]:
        return copy(self._attrs.outputs)

    def iterate(self):
        self._attrs = self._next_attrs
        self._next_attrs = copy(self._attrs)
        self._attrs.state = self._actions.next_state
        self._attrs.hidden_state = self._actions.next_hidden_state
        self._actions = None

    def add_input(self, other: Node, starting_strength: int = 0):
        if other not in self._next_attrs.inputs:
            self._next_attrs.inputs[other] = starting_strength
            other.add_output(self)

    def remove_input(self, other: Node):
        if other in self._next_attrs.inputs:
            del self._next_attrs.inputs[other]
            other.remove_output(self)

    def disconnect(self) -> None:
        for output in list(self._next_attrs.outputs):
            self.remove_output(output)
        for inp in list(self._next_attrs.inputs):
            self.remove_input(inp)

        del self

    def modify_connection_strength(self, other: Node, amount: float):
        self._next_attrs.inputs[other] += amount

    def get_connection_strength(self, other: Node):
        return self._next_attrs.inputs[other]

    def get_inputs(self) -> dict[Node, float]:
        return copy(self._attrs.inputs)

    def get_state(self) -> float:
        if isinstance(self, NeuronGenome):
            pass
        return self._attrs.state

    def is_excited(self) -> bool:
        return self._attrs.state >= FIRING_THRESHOLD

    def get_coords(self) -> Coords:
        return self._attrs.coords

    def move_to(self, coords: Coords):
        self._next_attrs.coords = coords

    def get_next_coords(self) -> Coords:
        return self._next_attrs.coords

    def add_reward(self, reward: Reward):
        self._next_attrs.rewards.append(reward)

    def add_cogni(self, amount):
        self._next_attrs.cogni += amount

    def sub_cogni(self, amount):
        self._next_attrs.cogni -= amount

    def get_units_moved(self) -> float:
        return get_distance(self._attrs.coords, self._next_attrs.coords)

    def distance_from(self, other: Node) -> Union[float, None]:
        if isinstance(other, CortexNode):
            return get_distance(self.get_coords(), other.get_coords())
        # Node does not exist on cortex.
        else:
            return None

    def get_surroundings(self) -> np.array:
        return self.surroundings

    def get_neuron_dist(self) -> int:
        return self.neuron_dist

    def set_surroundings(self, surroundings: List[float | int]) -> None:
        self.surroundings = np.array(surroundings[:-1])
        self.neuron_dist = int(surroundings[-1])

    def get_maintenance_value(self):
        return self._actions.maintenance_value

    def __hash__(self):
        return self.get_id()

    def __repr__(self):
        return (f"Neuron(id={self.get_id()}, "
                f"pos={list(self._attrs.coords)} "
                f"cogni={int(self.get_cogni())})")
