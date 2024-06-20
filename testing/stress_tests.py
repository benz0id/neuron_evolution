from time import sleep

from src.coordinators.phase_coordinator import ActionPhase
from testing.static_fixtures import get_managers


def stress():
    target_population = 100
    max_add = 1

    managers = get_managers()
    action_phase = ActionPhase(managers)

    for _ in range(target_population):
        managers.creator.spawn_neuron()

    print('here')

    action_phase.run_update()

    while True:
        print(f'Starting Epoch {managers.epoch_counter._get_epoch()}')
        action_phase.run_action_phase()
        action_phase.run_update()
        print(f'Finished Epoch {managers.epoch_counter._get_epoch()}')

        managers.creator.maintain_population(target_population, max_add)

        managers.epoch_counter.iterate()
        sleep(0.25)



