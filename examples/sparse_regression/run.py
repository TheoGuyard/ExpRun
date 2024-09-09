import numpy as np
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from experiment import Experiment, Runner


class SparseRegressionExperiment(Experiment):

    def setup(self):
        X, y = make_regression(**self.config["setup"])
        StandardScaler().fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        model = LassoCV().fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        return {"score": score}

    def cleanup(self):
        pass

    def plot(self, results):
        if len(results) == 0:
            return
        scores = []
        for result in results:
            scores.append(result["score"])
        print("Mean scores: ", np.mean(scores))


if __name__ == "__main__":

    exp_type = SparseRegressionExperiment
    current_dir = Path(__file__).parent
    config_path = current_dir.joinpath("config.yaml")
    results_dir = current_dir.joinpath("results")
    repeats = 10

    runner = Runner()
    # runner.run(exp_type, config_path, results_dir, repeats)
    runner.plot(exp_type, config_path, results_dir)
