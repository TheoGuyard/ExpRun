import pickle
import random
import string
import yaml
from abc import abstractmethod
from pathlib import Path
from typing import Union


class Experiment:

    def __init__(self, config_path: Union[str, Path]) -> None:
        config_path = Path(config_path)
        with open(config_path, "r") as file:
            self.config = yaml.load(file, Loader=yaml.Loader)

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def run(self) -> dict: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    @abstractmethod
    def plot(results: list) -> None: ...

    def save_result(self, result: dict, results_dir: Union[str, Path]) -> None:
        uuid_chars = string.ascii_lowercase
        output_uuid = "".join(random.choice(uuid_chars) for _ in range(20))
        output_name = "{}_{}.pkl".format(self.__class__.__name__, output_uuid)
        output_path = Path(results_dir, output_name)

        with open(output_path, "wb") as file:
            data = {"config": self.config, "result": result}
            pickle.dump(data, file)

    def load_result(self, file_path: Union[str, Path]):
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def find_results(self, results_dir: Union[str, Path]):
        matching_results = []
        result_pattern = "{}_*.pkl".format(self.__class__.__name__)
        for result_path in Path(results_dir).glob(result_pattern):
            result_data = self.load_result(result_path)
            if result_data["config"] == self.config:
                matching_results.append(result_data["result"])
        return matching_results

    def save_plot_data(
        self, plot_data: dict, plots_dir: Union[str, Path]
    ) -> None:
        uuid_chars = string.ascii_lowercase
        output_uuid = "".join(random.choice(uuid_chars) for _ in range(20))
        output_name = "{}_{}.pkl".format(self.__class__.__name__, output_uuid)
        output_path = Path(plots_dir, output_name)

        with open(output_path, "wb") as file:
            data = {"config": self.config, "plot_data": plot_data}
            pickle.dump(data, file)
