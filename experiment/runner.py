from pathlib import Path


class Runner:

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def run(
        self,
        experiment_type,
        config_path,
        results_dir,
        repeats=1,
    ):

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
        experiment_type,
        config_path,
        results_dir,
    ):

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
