from random import randint, uniform

import numpy as np

from src.entities.neuron import Actions


def get_random_actions() -> Actions:
    """
    Generates some random actions for a neuron.
    :return: Randomly generated actions.
    """
    return Actions(
        translation=            np.random.random_integers(-1, 1, 2),
        generosity=             uniform(0.01, 0.1),
        maintenance_value=      randint(0, 2),
        connection_willingness= randint(0, 13),
        next_state=             randint(0, 7),
        next_hidden_state=      np.random.random_integers(0, 7, 4),
        spawn_investment=       randint(1, 500),
        learning_rate=1
    )
