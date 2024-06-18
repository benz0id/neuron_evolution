from typing import List

from src.entities.neuron import Neuron, Reward, Node, get_distance
from src.entities.node_list import NodeList
from src.managers.cortex_manager import CortexManager
from src.managers.manager import Manager
from src.presenters.matrix_presenter import Presenter, PresenterSubject


class ConnectionManager(Manager, PresenterSubject):
    """
    Manages the health of the connections between neurons.

    Behaviours:

    The relationship strength between neurons decay by one each cycle.
    If a relationship reaches 0 that connection is lost.

    Neurons can invest cogni in a relationship to make it stronger.
    """

    cortex: CortexManager
    con_decay_rate: float
    con_start_value: int
    cogni_to_con_conversion: int
    dividend_scaling_factor: float
    connectivity_willingness_threshold: float
    presenters: List[Presenter]
    neurons: NodeList[Neuron]

    def __init__(self, con_decay_rate: float, con_start_value: int,
                 cogni_to_con_conversion: int, dividend_scaling_factor: float,
                 cortex_manager: CortexManager,
                 connectivity_willingness_threshold: float,
                 presenters: List[Presenter],
                 neurons: NodeList[Neuron]) -> None:
        Manager.__init__(self, Node)
        PresenterSubject.__init__(self, presenters)
        self.neurons = neurons
        self.con_decay_rate = con_decay_rate
        self.con_start_value = con_start_value
        self.cogni_to_con_conversion = cogni_to_con_conversion
        self.dividend_scaling_factor = dividend_scaling_factor
        self.cortex = cortex_manager
        self.connectivity_willingness_threshold = \
            connectivity_willingness_threshold
        self.presenters = presenters

    def _add(self, neuron: Neuron) -> None:
        pass

    def _remove(self, node: Neuron) -> None:
        pass

    def distribute_rewards(self, neuron: Neuron):
        """
        Redistributes all pending rewards currently assigned to the neuron.

        Amount that is redistrubuted depends on the generosity of the neuron.

        :param neuron:
        :return:
        """
        for record in neuron._to_distribute:
            for reward in record.rewards:
                if reward.distributed:
                    continue

                for input_neuron, state in record.input_states:
                    dividend = (state *
                                neuron._actions.generosity *
                                reward.cogni /
                                self.dividend_scaling_factor)
                    self.give_reward(neuron, input_neuron, dividend, )

    def give_reward(self, sender: Neuron, receiver: Node, cogni: float,
                    cur_distance: int = 0):
        """
        Gives a reward to a target neuron.

        :param sender: Neuron giving the reward.
        :param receiver: Neuron receiving the reward.
        :param cogni: Reward amount
        :param cur_distance: Propagation distance of reward.
        :return: None
        """

        self.handle_cogni_investment(sender, receiver, cogni)

        reward = Reward(sender, cogni, cur_distance + 1)
        self.update('reward', receiver)
        receiver.add_reward(reward)

    def add_new_random_connection(self, neuron: Neuron) -> bool:
        """
        Attempt to add a new connection to a neuron.
        :param neuron:
        :return:
        """
        if neuron.get_connection_willingness() < \
            self.connectivity_willingness_threshold:
            return False

        # Increase by two to account for return of this neuron and new
        # connection neuron.
        k = len(neuron.get_inputs()) + len(neuron.get_outputs()) + 2

        # Check that neuron isn't already connected to all neurons.
        if k >= len(self.neurons):
            return False

        nearest = self.cortex.get_k_nearest(neuron, k)
        nearest = sorted(nearest, key=lambda x: x[0])

        max_distance = neuron.get_max_radius()

        for distance, candidate in nearest:

            in_range = get_distance(neuron.get_coords(),
                                    candidate.get_coords()) < max_distance
            already_connected = \
                candidate in neuron.get_inputs() or \
                candidate in neuron.get_outputs() or \
                candidate is neuron

            if in_range and not already_connected:
                neuron.add_input(candidate, self.con_start_value)
                # self.handle_cogni_investment(neuron, candidate,
                #     TODO       forming connection is free?                 self.con_start_value)
                return True
        return False

    def do_connection_decay(self, neuron: Neuron) -> None:
        """
        Decay connections. Remove any connections that die.
        :param neuron:
        :return:
        """
        for input_node in neuron.get_inputs():
            neuron.modify_connection_strength(input_node, self.con_decay_rate)
            if neuron.get_connection_strength(input_node) < 0:
                neuron.remove_input(input_node)

    def maintain_relationships(self, neuron: Neuron) -> None:
        """
        Maintains each of the <neuron>'s input connections.
        :param neuron:
        :return:
        """
        investment = neuron.get_maintenance_value()
        for input_node in neuron.get_inputs():
            neuron.modify_connection_strength(input_node, investment)
            neuron.sub_cogni(investment)

    def handle_cogni_investment(self, neuron: Neuron, input_neuron: Node,
                                investment: float) -> None:
        """
        Handles the investment of cogni from <neuron> into its
        <input_neuron>.
        :param neuron: The neuron investing cogni.
        :param input_neuron: The neuron receiving cogni.
        :param investment: The amount of cogni invested.
        :return:
        """
        neuron.sub_cogni(investment)
        input_neuron.add_cogni(investment)
        added_connection_strength = investment * self.cogni_to_con_conversion
        neuron.modify_connection_strength(input_neuron, added_connection_strength)

    def prune_overextended_relationships(self, neuron: Neuron):
        """
        Remove
        :param neuron:
        :return:
        """
        for node in neuron.get_inputs():
            distance = neuron.distance_from(node)
            if distance is not None and distance > neuron.get_max_radius():
                neuron.remove_input(node)

        for node in neuron.get_outputs():
            distance = neuron.distance_from(node)
            if distance is not None and distance > neuron.get_max_radius():
                neuron.remove_output(node)



