import numpy as np
from pathlib import Path
from exprun import Experiment, Runner


class MyExperiment(Experiment):

    def setup(self) -> None:
        assert 0 < self.config["min"] <= self.config["max"]
        self.a = np.random.randint(self.config["min"], self.config["max"])
        self.b = np.random.randint(self.config["min"], self.config["max"])

    def run(self) -> dict:
        result = {
            "+": self.a + self.b,
            "*": self.a * self.b,
            "-": self.a - self.b,
            "/": self.a / self.b,
        }
        print(result)
        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list[dict]) -> None:
        keys = ["+", "*", "-", "/"]
        vals = {key: [] for key in keys}

        for result in results:
            for key in keys:
                vals[key].append(result[key])

        for key in keys:
            print(f"{key}: {np.mean(vals[key])}")


def test_experiment():
    config_path = Path(__file__).parent.joinpath("test_experiment.yml")
    result_dir = Path(__file__).parent.joinpath("results")
    save_dir = Path(__file__).parent.joinpath("saves")
    repeats = 5
    verbose = True

    runner = Runner(verbose=verbose)
    runner.run(MyExperiment, config_path, result_dir, repeats)
    runner.plot(MyExperiment, config_path, result_dir, save_dir)
