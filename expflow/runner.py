from pathlib import Path
from typing import Union
from .experiment import Experiment


class Runner:

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def run(
        self,
        experiment_type: Experiment,
        config_path: Path,
        results_dir: Path,
        repeats: int = 1,
    ) -> None:

        assert issubclass(experiment_type, Experiment)
        assert isinstance(config_path, Path)
        assert isinstance(results_dir, Path)
        assert isinstance(repeats, int)
        assert repeats >= 1

        config_path = Path(config_path)

        if self.verbose:
            print("")
            print("experiment: {}".format(experiment_type.__name__))
            print("config file: {}".format(config_path.name))
            print("results dir: {}".format(results_dir))

        experiment = experiment_type(config_path)
        for repeat in range(repeats):
            if self.verbose:
                print()
                print("repeat {}/{}".format(repeat + 1, repeats))
            experiment.setup()
            result = experiment.run()
            experiment.save_result(result, results_dir)
            experiment.cleanup()

    def plot(
        self,
        experiment_type: Experiment,
        config_path: Union[str, Path],
        results_dir: Union[str, Path],
        save_dir: Union[str, Path, None] = None,
    ) -> None:

        assert issubclass(experiment_type, Experiment)
        assert isinstance(config_path, Path)
        assert isinstance(results_dir, Path)

        config_path = Path(config_path)

        if self.verbose:
            print("experiment: {}".format(experiment_type.__name__))
            print("config file: {}".format(config_path.name))
            print("results dir: {}".format(results_dir))
            print("save dir: {}".format(save_dir))
            print("searching results...")

        experiment = experiment_type(config_path)
        results = experiment.find_results(results_dir)

        if self.verbose:
            print("  found: {} matching results".format(len(results)))

        if len(results) == 0:
            return

        if self.verbose:
            print("plotting...")

        self.plot_data = experiment.plot(results)

        if save_dir is not None:
            if self.verbose:
                print("saving...")
            experiment.save_plot(self.plot_data, save_dir)
