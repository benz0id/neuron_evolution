import numpy as np
from entities.neuron import Actions, Neuron

from testing.static_fixtures import get_managers


def test_failed_mating_pair():
    managers = get_managers()

    managers.creator.breeding_threshold = 20
    managers.creator.partner_search_radius = 10

    neurons = managers.nodes.get_all(Neuron)

    n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
    n2 = managers.creator.spawn_neuron(np.array([42, 16]), False)

    managers.cortex.refresh_tree()
    managers.genomes.classify_species()

    a = Actions(
        translation=np.array([0, 0]),
        generosity=0,
        maintenance_value=0,
        connection_willingness=0,
        next_state=0,
        next_hidden_state=0,
        spawn_investment=10,
        learning_rate=1
    )

    n1.set_actions(a)
    n2.set_actions(a)

    managers.creator.breed_willing_neurons(neurons)

    assert len(neurons) == 2

def test_successful_mating_pair():
    ss = False
    while not ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 10

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
        n2 = managers.creator.spawn_neuron(np.array([23, 16]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if not ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=40,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 3

def test_succ_mating_pair_distance():
    ss = False
    while not ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 10

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
        n2 = managers.creator.spawn_neuron(np.array([32, 16]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if not ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 3

def test_fail_mating_pair_distance():
    ss = False
    while not ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 10

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
        n2 = managers.creator.spawn_neuron(np.array([33, 16]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if not ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 4

def test_fail_mating_pair_distance_diag():
    ss = False
    while not ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 200 ** (1/2) - 1

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([10, 10]), False)
        n2 = managers.creator.spawn_neuron(np.array([20, 20]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if not ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 4

def test_succ_mating_pair_distance_diag():
    ss = False
    while not ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 200 ** (1/2) + 1

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([10, 10]), False)
        n2 = managers.creator.spawn_neuron(np.array([20, 20]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if not ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)
        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 3


def test_fail_mating_pair_species():
    ss = True
    while ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 10

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
        n2 = managers.creator.spawn_neuron(np.array([33, 16]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 4

def test_fail_mating_pair_cogni():
    ss = True
    while ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 30
        managers.creator.partner_search_radius = 10

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
        n2 = managers.creator.spawn_neuron(np.array([33, 16]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        assert len(neurons) == 2

def test_cogni_balances():
    ss = False
    while not ss:
        managers = get_managers()

        managers.creator.breeding_threshold = 20
        managers.creator.partner_search_radius = 10

        neurons = managers.nodes.get_all(Neuron)

        n1 = managers.creator.spawn_neuron(np.array([22, 16]), False)
        n2 = managers.creator.spawn_neuron(np.array([32, 16]), False)

        managers.genomes.classify_species()

        ss = n1._genome.species == n2._genome.species

        if not ss:
            continue

        a = Actions(
            translation=np.array([0, 0]),
            generosity=0,
            maintenance_value=0,
            connection_willingness=0,
            next_state=0,
            next_hidden_state=0,
            spawn_investment=21,
            learning_rate=1
        )

        n1.set_actions(a)
        n2.set_actions(a)

        managers.creator.breed_willing_neurons(neurons)

        for neuron in neurons:
            neuron.iterate()

        assert len(neurons) == 3
        assert n1.get_cogni() == 79
        assert n2.get_cogni() == 79
        assert neurons[2].get_cogni() == 42
