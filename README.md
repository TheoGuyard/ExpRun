# ExpRun

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyPI version](https://badge.fury.io/py/exprun.svg)](https://pypi.org/project/exprun/)
[![codecov](https://codecov.io/gh/TheoGuyard/ExpRun/graph/badge.svg?token=yqikBSTySk)](https://codecov.io/gh/TheoGuyard/ExpRun)
[![Test Status](https://github.com/TheoGuyard/exprun/actions/workflows/test.yml/badge.svg)](https://github.com/TheoGuyard/exprun/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/TheoGuyard/ExpRun/blob/main/LICENSE)

*Automatized python pipelines to run reproducible experiments.*

## Installation

`exprun` is available on [pypi](https://pypi.org/project/exprun). The latest version of the package can be installed as


```shell
pip install exprun
```

## Quick start

With `exprun`, an experiment is run using an instance of the `Runner` class, based on a [yaml](https://yaml.org) configuration file, with a specified results directory, save directory, and number of repeats.
The user is only required to fill the four methods in inherited from the `Experiment` class.

Here is a simple example of experiment that computes the sum of two random numbers drawn uniformly over a specified range.
The configuration file is as follows.

```yaml
# file: config.yml

min: 1
max: 10
```

The experiment can then be created is as follows.

```python
# file: myexperiment.py

from exprun import Experiment, Runner

class MyExperiment(Experiment):
    def setup(self) -> None:
        # Set up the experiment from the information in the configuration file.
        ...

    def run(self) -> dict:
        # Perform one run of the experiment and return the results as a dict.
        ...

    def cleanup(self) -> None:
         # Clean up the experimental data.
        ...

    def plot(self, results: list) -> dict:
         # Process the results obtained and returns the plot data that must be saved.
        ...

config_path = './config.yml'
result_dir = './results/'
save_dir = './saves/'

runner = Runner()
runner.run(MyExperiment, config_path, result_dir, repeats=10)
runner.plot(MyExperiment, config_path, result_dir, save_dir)
```

Launching `python myexperiment.py` runs 10 times `MyExperiment`, each result is saved at `result_dir` and the plot data is saved as a [pickle](https://docs.python.org/3/library/pickle.html) file.
Then, the results found in the `result_dir` that match the current configuration are recovered, plotted, and saved at `save_dir` as a [pickle](https://docs.python.org/3/library/pickle.html) file.
And as simple as that, you have a reproducible experiment that can be run multiple times with different configurations.
More advanced examples can be found in the [examples](https://github.com/TheoGuyard/ExpRun/tree/main/examples) directory.

## Contribute

`exprun` is still in its early stages of development.
Feel free to contribute by reporting any bugs on the [issue](https://github.com/TheoGuyard/ExpRun/issues) page or by opening a [pull request](https://github.com/TheoGuyard/ExpRun/pulls).
Any feedback or contribution is welcome.

## License

`exprun` is distributed under the
[MIT](https://github.com/TheoGuyard/ExpRun/blob/main/LICENSE) license.
