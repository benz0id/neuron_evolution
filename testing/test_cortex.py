import numpy as np

from testing.static_fixtures import get_managers

vec = np.array

def test_sectors_fully_surrounded() -> None:
    managers = get_managers()

    managers.cortex.do_sector_distance_scaling = False

    center = managers.creator.spawn_neuron(vec([12, 12]))

    assert managers.cortex.get_surrounding_densities(center) == [0, 0, 0, 0, 7]


    right = managers.creator.spawn_neuron(vec([20, 12]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) == [1, 0, 0, 0, 1]
    left = managers.creator.spawn_neuron(vec([4, 12]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) == [1, 1, 0, 0, 1]

    top = managers.creator.spawn_neuron(vec([12, 20]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) \
           == [1, 1, 1, 0, 1]

    bottom = managers.creator.spawn_neuron(vec([12, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False)\
           == [1, 1, 1, 1, 1]

def test_sectors_iterative() -> None:
    managers = get_managers()

    managers.cortex.do_sector_distance_scaling = False

    center = managers.creator.spawn_neuron(vec([12, 12]))

    assert managers.cortex.get_surrounding_densities(center) == [0, 0, 0, 0, 7]


    right = managers.creator.spawn_neuron(vec([20, 12]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) == [1, 0, 0, 0, 1]
    managers.nodes.remove_node(right)
    left = managers.creator.spawn_neuron(vec([4, 12]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) == [0, 1, 0, 0, 1]
    managers.nodes.remove_node(left)

    top = managers.creator.spawn_neuron(vec([12, 20]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) \
           == [0, 0, 1, 0, 1]
    managers.nodes.remove_node(top)

    bottom = managers.creator.spawn_neuron(vec([12, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False)\
           == [0, 0, 0, 1, 1]

def test_sectors_cornered() -> None:
    managers = get_managers()

    managers.cortex.do_sector_distance_scaling = False

    center = managers.creator.spawn_neuron(vec([12, 12]))

    tl = managers.creator.spawn_neuron(vec([4, 20]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False)\
           == [0, 1, 1, 0, 1]
    bl = managers.creator.spawn_neuron(vec([4, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False)\
           == [0, 2, 1, 1, 1]

    tr = managers.creator.spawn_neuron(vec([20, 20]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) \
           == [1, 2, 2, 1, 1]

    br = managers.creator.spawn_neuron(vec([20, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False)\
           == [2, 2, 2, 2, 1]

def test_sectors_distal() -> None:
    managers = get_managers()

    managers.cortex.do_sector_distance_scaling = False

    center = managers.creator.spawn_neuron(vec([4, 4]))

    other = managers.creator.spawn_neuron(vec([12, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False)\
           == [1, 0, 0, 0, 1]
    managers.nodes.remove_node(other)

    other = managers.creator.spawn_neuron(vec([20, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) \
           == [1, 0, 0, 0, 2]
    managers.nodes.remove_node(other)

    other = managers.creator.spawn_neuron(vec([28, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) \
           == [1, 0, 0, 0, 3]
    managers.nodes.remove_node(other)

    other = managers.creator.spawn_neuron(vec([36, 4]))
    assert managers.cortex.get_surrounding_densities(center, avoid_rerun=False) \
           == [1, 0, 0, 0, 4]
    managers.nodes.remove_node(other)

