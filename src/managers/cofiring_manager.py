from typing import Callable, List

from src.entities.neuron import Neuron, Reward, CofireRecord
from src.managers.manager import Manager
from src.presenters.matrix_presenter import Presenter, PresenterSubject


class CofiringManager(Manager, PresenterSubject):

    """
    Handles the cofiring of neurons.
    """

    cofire_expiry_age: int
    get_epoch: Callable[[], int]
    presenters: List[Presenter]

    def __init__(self, cofire_expiry_age: int, get_epoch: Callable[[], int],
                 presenters: List[Presenter]):
        Manager.__init__(self, [Neuron])
        PresenterSubject.__init__(self, presenters)
        self.cofire_expiry_age = cofire_expiry_age
        self.get_epoch = get_epoch
        self.presenters = presenters

    def _add(self, neuron: Neuron) -> None:
        pass

    def _remove(self, node: Neuron) -> None:
        pass

    def create_input_fire_record(self, neuron: Neuron):
        """
        Stores the current state of the inputs.
        :param neuron:
        :return:
        """

        if not neuron.get_inputs():
            neuron.set_input_state_record(None)

        record = CofireRecord(
            self.get_epoch(),
            [],
        )

        any_activity = False
        for input in neuron.get_inputs():
            if input.is_excited():
                record.input_states.append(
                    (input, input.get_state())
                )
                any_activity = True

        if not any_activity:
            neuron.set_input_state_record(None)
            return

        neuron.set_input_state_record(record)

    def create_firing_record(self, neuron: Neuron):
        """
        If the previous round of inputs caused the neuron to fire, upgrade the
        input record to a canadidate firing record.
        :param neuron:
        :return:
        """
        if neuron.is_excited():
            self.update('fire', neuron)
            record = neuron.get_last_input_state()
            neuron.add_candidiate_record(record)
        else:
            neuron.del_candidiate_record()

    def complete_cofiring_record(self, neuron: Neuron):
        """
        If the candidate firing record appears to have lead  to the firing of an
        output neuron, save that record as a cofiring record.
        """

        # If no inputs fired last round, or the neuron has no outputs then
        # continue.
        if neuron.get_candidiate_record() is None or not neuron.get_outputs():
            return

        firing_record = neuron.get_candidiate_record()
        assert not firing_record.output_found

        for output in neuron.get_outputs():
            if output.is_excited():
                firing_record.resultant_output_states[output] = \
                output.get_state()
                firing_record.output_found = True

        neuron.add_cofire_record(firing_record)

    def prune_cofires(self, neuron: Neuron) -> None:
        """
        Update the age of all cofires and prune them if they have expired.
        """
        for record in neuron._cofiring_records:
            record.age += 1

        if neuron._cofiring_records[-1].age > self.cofire_expiry_age:
            neuron._cofiring_records.pop()

    def attribute_reward(self, neuron: Neuron, reward: Reward) -> None:
        """
        Assigns the reward to a cofiring event within the receiving neuron.

        :param neuron: The neuron receiving the reward.
        :param reward: The reward to be assigned to a cofiring event.
        :return: None
        """
        last_distance = reward.distance
        best_record = None

        new_distance = last_distance - 1
        i = 0

        while new_distance < last_distance and i < len(neuron._cofiring_records):
            last_distance = new_distance
            new_distance = reward.distance * 2 - neuron._cofiring_records[i].age

            if reward.sender in neuron.get_cofire_records()[i].resultant_output_states:
                best_record = neuron.get_cofire_records()[i]

        if reward is not None:
            best_record.rewards.append(reward)
            neuron._to_distribute.add(best_record)

        # TODO handle case where reward cannot be attributed to a firing event.






