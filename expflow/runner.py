from pathlib import Path
from typing import Union
from .experiment import Experiment


class Runner:

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def run(
        self,
        experiment_type: Experiment,
        config_path: Union[str, Path],
        results_dir: Union[str, Path],
        repeats: int = 1,
    ) -> None:

        assert repeats >= 1

        config_path = Path(config_path)

        if self.verbose:
            print("config file: {}".format(config_path.name))

        experiment = experiment_type(config_path)
        for repeat in range(repeats):
            if self.verbose:
                print("repeat {}/{}".format(repeat + 1, repeats))
                print("  running...")
            experiment.setup()
            result = experiment.run()
            experiment.save_result(result, results_dir)
            experiment.cleanup()

    def plot(
        self,
        experiment_type: Experiment,
        config_path: Union[str, Path],
        results_dir: Union[str, Path],
    ) -> None:

        assert isinstance(experiment_type, Experiment)
        assert isinstance(config_path, Path)
        assert isinstance(results_dir, Path)

        config_path = Path(config_path)

        if self.verbose:
            print("config file: {}".format(config_path.name))
            print("searching results...")

        experiment = experiment_type(config_path)
        results = experiment.find_results(results_dir)

        if self.verbose:
            print("  found: {}".format(len(results)))

        if len(results) == 0:
            return

        if self.verbose:
            print("plotting...")
        experiment.plot(results)
