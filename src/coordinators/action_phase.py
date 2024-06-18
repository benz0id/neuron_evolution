from src.entities.neuron import Neuron
from src.managers.epoch_counter import EndOfSimulation
from src.managers.helpers.get_random_actions import get_random_actions
from src.managers.helpers.manager_set import ManagerSet


class ActionPhase:
    """
    Responsible for orchestrating the action phase of the simulation. This
    is parrelelizable, such that each neuron can compute and act on its actions
    independently.

    0. Compute Actions

    Compute the actions that will be taken by each neuron. This will become more
    computaitonally intensive as the complexity of the neural genomes increases.


    1. Change Position

        Neurons may move around on the board.

        Note. Neurons cannot move to a location where another neuron currently
        is, but may overlap if the both move to that location. #TODO fix.

    2. Maintain Relationships

        Neurons may strengthen their relationships with other neurons.

    3. Redistribute Rewards

        If any rewards were received during the last round, neurons may
        redistribute them to their inputs as they see fit.

    3. Erode and Prune Relationships

        Input connection strength on all neurons are reduced by a fixed amount.
        Connections that have reached a strength of zero are pruned.

    5. Form Relationships

        Attempt to form relationships with other nearby neurons.

    6. Subtract Costs

        Subtract basic costs of operating from the neuron. If it is out of cogni
        then remove it from the simulation.

    7. Spawn Progeny.
    """

    managers: ManagerSet

    def __init__(self, managers: ManagerSet):
        self.managers = managers

    def run_action_phase(self) -> None:
        """
        :param neuron:
        :return:
        """

        neurons = self.managers.nodes.get_all(Neuron)

        # # Gather local neuron density attribute.
        for neuron in neurons:
            density_input = self.managers.cortex.get_surrounding_densities(neuron)
            pass

        # Compute neuron actions TODO using genome.
        for neuron in neurons:
            actions = get_random_actions()
            neuron.set_actions(actions)

        # Remove relationships that are out of range.
        for neuron in neurons:
            self.managers.connections.prune_overextended_relationships(neuron)

        # TODO Temporary - belongs in learning phase.
        for neuron in neurons:
            self.managers.firing.create_firing_record(neuron)

        # Move neuron.
        for neuron in neurons:
            self.managers.cortex.move_node(neuron, neuron.get_translation())

        # Invest in relationships.
        for neuron in neurons:
            self.managers.connections.maintain_relationships(neuron)

        # Redistribute rewards.
        for neuron in neurons:
            self.managers.connections.distribute_rewards(neuron)

        # Erode relationships
        for neuron in neurons:
            self.managers.connections.do_connection_decay(neuron)

        # Form connections if the neuron is willing.
        for neuron in neurons:
            self.managers.connections.add_new_random_connection(neuron)

        # Subtract basic costs from neurons.
        self.managers.costs.do_cost_subtraction()

        # Remove depleted neurons from the simulation.
        self.managers.costs.prune_depleted_neurons()

        # Breed the willing neurons.
        self.managers.creator.breed_willing_neurons(neurons)

        # If required, add new neurons to maintiain population.
        self.managers.creator.maintain_population()

    def run_update(self) -> None:
        """
        To be run after the action phase.
        Iterate neurons and update manager data structures to new neuron
        states.
        :return:
        """

        self.managers.analysis.store_simulation_stats()

        neurons = self.managers.nodes.get_all(Neuron)

        try:
            self.managers.epoch_counter.iterate()
        except EndOfSimulation as e:
            self.managers.analysis.compile_figures()
            raise e

        for neuron in neurons:
            neuron.iterate()

        self.managers.genomes.classify_species()
        self.managers.cortex.refresh_tree()








