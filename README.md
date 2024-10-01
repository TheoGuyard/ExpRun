# ExpFlow

`expflow` enables building simple pipelines for reproducible numerical experiments.

## Quick start

An experiment is run using an instance of the `Runner` class, based on a [YAML](https://yaml.org) configuration file, with a specified results directory and number of repeats.

```python
from expflow import Experiment, Runner

class MyExperiment(Experiment):
    ...

config_path = './config.yml'
results_dir = './results/'

runner = Runner()
runner.run(MyExperiment, config_path, results_dir, repeats=10)
runner.plot(MyExperiment, config_path, results_dir)
```

With the above code, `MyExperiment` is run 10 times, each result is saved in the specified `results_dir` directory as a [pickle](https://docs.python.org/3/library/pickle.html) file and results found in this directory that match the current configuration are recovered and plotted.

In the `MyExperiment` class, four functions need to be specified:

- `setup(self) -> None`: Set up the experiment from the information in the configuration file.
- `run(self) -> dict`: Perform one run of the experiment and return the results as a dict.
- `cleanup(self) -> None`: Clean up the experimental data.
- `plot(self, results: list[dict]) -> None`: Process the results obtained from a list of all the results found in the results saving directory that match the configuration file.

For instance, a simple experiment that computes the sum of two numbers within a range specified in the configuration file could be defined as follows.
First, let's write the `config.yml` file.

```yaml
min: 1
max: 10
```

Next, let's fill the `MyExperiment` class functions.

```python
class MyExperiment(Experiment):

    def setup(self) -> None:
        from random import randrange
        # You can access the config data from self.config
        self.a = randrange(self.config["min"], self.config["max"])
        self.b = randrange(self.config["min"], self.config["max"])
    
    def run(self) -> dict:
        # You can access the data of the setup function
        return {"sum": self.a + self.b}

    def cleanup(self) -> None:
        # Nothing to clean up here
        pass

    def plot(self, results: list[dict]) -> None:
        # Print the mean of the sums among all the results
        # found matching the current configuration file
        sums = [result["sum"] for result in results]
        print("Mean of sums:", sum(sums) / len(sums))
```

And as simple as that, you have a reproducible experiment that can be run multiple times with different configurations.
More advanced examples can be found in the [examples](https://github.com/TheoGuyard/ExpFlow/tree/main/examples) directory.

## Contribute

`expflow` is still in its early stages of development.
Feel free to contribute by reporting any bugs on the [issue](https://github.com/TheoGuyard/ExpFlow/issues) page or by opening a [pull request](https://github.com/TheoGuyard/ExpFlow/pulls).
Any feedback or contribution is welcome.

## License

`expflow` is distributed under the
[MIT](https://github.com/TheoGuyard/ExpFlow/blob/main/LICENSE) license.