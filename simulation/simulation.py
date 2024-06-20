from multiprocessing import freeze_support
from time import sleep
from timeit import default_timer as timer
from simulation_fixtures import get_managers
from src.controllers.phase_coordinator import PhaseController


def main():

    managers = get_managers()
    controller = PhaseController(managers)
    controller.run_setup()

    terminated = False
    last = timer()

    while not terminated:
        controller.run_action_phase()

        epoch = managers.epoch_counter.get_epoch()
        neuron_count = managers.nodes.get_neuron_count()
        now = timer()
        time = now - last
        last = now
        print(f'Epoch \t #{epoch}\t{neuron_count}\t{time:.2f}s')

        terminated = not controller.run_update()


if __name__ == '__main__':
    freeze_support()
    main()



