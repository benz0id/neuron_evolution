from time import sleep

from simulation_fixtures import get_managers
from src.coordinators.action_phase import ActionPhase
from src.coordinators.setup_phase import SetupPhase


def main():

    managers = get_managers()
    action_phase = ActionPhase(managers)
    setup_phase = SetupPhase(managers)

    setup_phase.run_setup()
    while True:
        action_phase.run_action_phase()
        action_phase.run_update()

        epoch = managers.epoch_counter.get_epoch()
        neuron_count = managers.nodes.get_neuron_count()
        print(f'Epoch \t #{epoch}\t{neuron_count}')


main()



