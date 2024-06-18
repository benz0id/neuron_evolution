import numpy as np
from entities.neuron import Actions

from testing.static_fixtures import get_managers


def test_connecting_pair():
    managers = get_managers()


    n1 = managers.creator.spawn_neuron(np.array([10, 10]), False)
    n2 = managers.creator.spawn_neuron(np.array([11, 11]), False)

    neurons = [n1, n2]
    for neuron in neurons:
        neuron._actions.connection_willingness = 10

    managers.cortex.refresh_tree()

    managers.connections.add_new_random_connection(n1)

    assert n2 in n1._next_attrs.inputs
    assert n1 in n2._next_attrs.outputs

    n1.remove_input(n2)

    assert n2 not in n1._next_attrs.inputs
    assert n1 not in n2._next_attrs.outputs


def test_failed_connecting_pair():
    managers = get_managers()

    n1 = managers.creator.spawn_neuron(np.array([10, 10]), False)
    n2 = managers.creator.spawn_neuron(np.array([21, 10]), False)

    neurons = [n1, n2]
    for neuron in neurons:
        neuron._actions.connection_willingness = 10

    managers.cortex.refresh_tree()

    managers.connections.add_new_random_connection(n1)

    assert n2 not in n1._next_attrs.inputs
    assert n1 not in n2._next_attrs.outputs

def test_reward_propogation_and_firing():
    managers = get_managers()

    n0 = managers.creator.spawn_neuron(np.array([10, 10]), False)
    n1 = managers.creator.spawn_neuron(np.array([18, 10]), False)
    n2 = managers.creator.spawn_neuron(np.array([23, 10]), False)

    neurons = [n0, n1, n2]
    for neuron in neurons:
        neuron._actions.connection_willingness = 10
        neuron._actions.generosity = 1

    def iterate():
        for n in neurons:
            n.iterate()

    managers.cortex.refresh_tree()

    assert managers.connections.add_new_random_connection(n0)
    assert managers.connections.add_new_random_connection(n1)

    iterate()

    assert n1 in n0._next_attrs.inputs
    assert n2 in n1._next_attrs.inputs

    n2._attrs.state = 10
    managers.firing.create_input_fire_record(n1)

    iterate()

    n1._attrs.state = 10

    managers.firing.create_firing_record(n1)

    iterate()

    n0._attrs.state = 10

    managers.firing.complete_cofiring_record(n1)

    managers.connections.give_reward(n0, n1, 100)

    iterate()

    assert n0.get_cogni() == 0
    assert n1.get_cogni() == 200

    n1._actions = Actions(np.array([0, 0]), 0.2, 0,  0, 0, 0, 0, 0)

    managers.firing.attribute_reward(n1, n1.get_rewards()[0])

    managers.connections.distribute_rewards(n1)

    iterate()

    assert n0.get_cogni() == 0
    assert n1.get_cogni() == 100
    assert n2.get_cogni() == 200


