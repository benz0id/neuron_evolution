from src.managers.helpers.manager_set import ManagerSet


class SetupPhase:

    """
    Configures managers and environment during the 0th epoch.
    """

    def __init__(self, managers: ManagerSet):
        self.managers = managers

    def run_setup(self) -> None:
        self.managers.creator.spawn_initial_population()

        self.managers.genomes.classify_species()

        self.managers.analysis.store_simulation_stats()

        self.managers.epoch_counter.iterate()

        self.managers.cortex.refresh_tree()




