import numpy as np

from testing.static_fixtures import get_managers

managers = get_managers()

for i in range(16):
    neuron = managers.creator.spawn_neuron(np.array([i, 0]), False)
    neuron._genome.species = i




