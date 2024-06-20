from time import sleep

from src.analysis.timer import PipelineTimer
from src.entities.neuron import Neuron
from src.managers.epoch_counter import EndOfSimulation
from src.managers.helpers.get_random_actions import get_random_actions
from src.managers.helpers.manager_set import ManagerSet
from multiprocessing import Pool


class PhaseController:


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

    def run_setup(self) -> None:
        self.managers.creator.spawn_initial_population()

        self.managers.genomes.classify_species()

        self.managers.analysis.store_simulation_stats()

        self.managers.epoch_counter.iterate()

        self.managers.cortex.refresh_tree()

    def run_action_phase(self) -> None:
        """
        :param neuron:
        :return:
        """
        timer = self.managers.action_timer

        neurons = self.managers.nodes.get_all(Neuron)

        # Gather local neuron density.
        timer.begin('local_density_calculation')
        for neuron in neurons:
            neuron.set_surroundings(
                self.managers.cortex.get_surrounding_densities(neuron))

        # Compute neuron actions.
        self.managers.genomes.update_all_actions(timer)

        # Remove relationships that are out of range.
        timer.begin('prune_distal_connections')
        for neuron in neurons:
            self.managers.connections.prune_overextended_relationships(neuron)

        # TODO Temporary - belongs in learning phase.
        timer.begin('create_firing_records')
        for neuron in neurons:
            self.managers.firing.create_firing_record(neuron)

        # Move neurons.
        timer.begin('move_neurons')
        for neuron in neurons:
            self.managers.cortex.move_node(neuron, neuron.get_translation())

        # Invest in relationships.
        timer.begin('relation_investment')
        for neuron in neurons:
            self.managers.connections.maintain_relationships(neuron)

        # Redistribute rewards.
        timer.begin('reward_distribution')
        for neuron in neurons:
            self.managers.connections.distribute_rewards(neuron)

        # Erode relationships
        timer.begin('relationship_erosion')
        for neuron in neurons:
            self.managers.connections.do_connection_decay(neuron)

        # Form connections if the neuron is willing.
        timer.begin('form_connections')
        for neuron in neurons:
            self.managers.connections.add_new_random_connection(neuron)

        # Subtract basic costs from neurons.
        timer.begin('subtract_costs')
        self.managers.costs.do_cost_subtraction()

        # Remove depleted neurons from the simulation.
        timer.begin('kill_neurons')
        self.managers.costs.prune_depleted_neurons()

        # Breed the willing neurons.
        timer.begin('breed_neurons')
        self.managers.creator.breed_willing_neurons(neurons)

        # If required, add new neurons to maintiain population.
        timer.begin('population_maintenance')
        self.managers.creator.maintain_population()
        timer.end()

    def run_update(self) -> bool:
        """
        To be run after the action phase.
        Iterate neurons and update manager data structures to new neuron
        states.
        :return:
        """
        timer = self.managers.update_timer

        timer.begin('store_stats')
        self.managers.analysis.store_simulation_stats()
        timer.end()

        neurons = self.managers.nodes.get_all(Neuron)

        for neuron in neurons:
            neuron.iterate()

        timer.begin('speciation')
        self.managers.genomes.classify_species()

        timer.begin('build_search_tree')
        self.managers.cortex.refresh_tree()
        timer.end()

        self.managers.cortex.refresh_sectors()

        self.managers.update_timer.iterate()
        self.managers.action_timer.iterate()

        try:
            self.managers.epoch_counter.iterate()
        except EndOfSimulation as e:
            self.managers.analysis.compile_figures()
            print(e.__str__())
            return False
        return True









