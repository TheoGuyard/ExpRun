import numpy as np
from pathlib import Path
from expflow import Experiment


class TestExperiment(Experiment):

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
    config_path = Path(__file__).parent.joinpath("test_config.yml")
    experiment = TestExperiment(config_path)
    experiment.setup()
    result = experiment.run()
    experiment.cleanup()

    assert list(result.keys()) == ["+", "*", "-", "/"]
    assert result["+"] >= 2 * experiment.config["min"]
    assert result["+"] <= 2 * experiment.config["max"]
    assert result["*"] >= experiment.config["min"] ** 2
    assert result["*"] <= experiment.config["max"] ** 2
    assert result["-"] >= experiment.config["min"] - experiment.config["max"]
    assert result["-"] <= experiment.config["max"] - experiment.config["min"]
    assert result["/"] >= experiment.config["min"] / experiment.config["max"]
    assert result["/"] <= experiment.config["max"] / experiment.config["min"]
