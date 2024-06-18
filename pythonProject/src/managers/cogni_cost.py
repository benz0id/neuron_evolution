from typing import Callable

from src.entities.neuron import Neuron
from src.entities.node import Node
from src.entities.node_list import NodeList
from src.managers.manager import Manager


class CogniCostManager(Manager):
    """
    Deducts cogni from neurons based on the actions they took during the past
    cycle.
    """

    neurons: NodeList[Neuron]

    clear_neuron: Callable[[Neuron], None]

    input_cost: float
    existence_cost: float
    state_cost: float
    movement_cost: float

    def __init__(self, neurons: NodeList[Neuron],
                 clear_neuron: Callable[[Neuron], None],
                 relationship_cost: float,
                 existence_cost: float,
                 state_cost: float,
                 movement_cost: float):
        """

        :param neurons: All neurons.
        :param clear_neuron: Used to eliminate neurons that have run out of
            cogni.
        :param relationship_cost: Cogni per relationship per round.
        :param existence_cost: Cogni subtracted per round.
        :param state_cost: The cost per level of active state.
        :param movement_cost: The cost per unit area moved.
        """
        super().__init__(Neuron)
        self.clear_neuron = clear_neuron
        self.neurons = neurons
        self.input_cost = relationship_cost
        self.existence_cost = existence_cost
        self.state_cost = state_cost
        self.movement_cost = movement_cost

    def do_cost_subtraction(self) -> None:
        for neuron in self.neurons:
            self.subtract_costs(neuron)

    def prune_depleted_neurons(self):
        """
        Removes all neurons that died this cycle.
        :return:
        """
        i = 0
        while i < len(self.neurons):
            neuron = self.neurons[i]
            if neuron.get_next_cogni() < 0:
                self.clear_neuron(neuron)
            else:
                i += 1

    def subtract_costs(self, neuron: Neuron):
        cost = self.existence_cost + \
            neuron.get_state() * self.state_cost + \
            len(neuron.get_inputs()) * self.input_cost + \
            neuron.get_units_moved() * self.movement_cost
        neuron.sub_cogni(cost)

    def _remove(self, node: Node) -> None:
        pass

    def _add(self, node: Node) -> None:
        pass
