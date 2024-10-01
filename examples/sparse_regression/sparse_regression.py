import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import (
    ElasticNetCV,
    LarsCV,
    LassoCV,
    OrthogonalMatchingPursuitCV,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from expflow import Experiment, Runner


class SparseRegression(Experiment):

    bindings = {
        "elastic-net": ElasticNetCV(),
        "lars": LarsCV(),
        "lasso": LassoCV(),
        "omp": OrthogonalMatchingPursuitCV(),
        "ridge": RidgeCV(),
    }

    def setup(self) -> None:
        X, y, coef = make_regression(coef=True, **self.config["dataset"])
        X = StandardScaler().fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **self.config["split"]
        )
        self.coef = coef
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self) -> dict:
        result = {}
        for model_name in self.config["model_names"]:
            result[model_name] = {}
            if model_name not in self.bindings.keys():
                raise ValueError(f"Unknown model {model_name}.")
            model = self.bindings[model_name]
            model.fit(self.X_train, self.y_train)
            coef_supp = self.coef != 0
            model_coef_supp = model.coef_ != 0
            y_train_pred = np.dot(self.X_train, model.coef_)
            y_test_pred = np.dot(self.X_test, model.coef_)
            result[model_name]["train_error"] = mean_squared_error(
                self.y_train, y_train_pred
            )
            result[model_name]["test_error"] = mean_squared_error(
                self.y_test, y_test_pred
            )
            result[model_name]["f1_score"] = f1_score(
                coef_supp, model_coef_supp
            )

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list[dict]) -> None:
        stats = {
            model_name: {"train_error": [], "test_error": [], "f1_score": []}
            for model_name in self.config["model_names"]
        }

        for result in results:
            for model_name, model_stats in result.items():
                for stat_name, stat_value in model_stats.items():
                    stats[model_name][stat_name].append(stat_value)

        mean_stats = {
            model_name: {
                stat_name: np.mean(stat_values)
                for stat_name, stat_values in model_stats.items()
            }
            for model_name, model_stats in stats.items()
        }
        metric_names = mean_stats[next(iter(mean_stats))].keys()
        max_stats = {
            metric: max(mean_stats[model][metric] for model in mean_stats)
            for metric in metric_names
        }

        model_names = list(mean_stats.keys())
        bar_width = 0.1
        index = np.arange(len(metric_names))
        offsets = np.linspace(
            -bar_width * len(model_names) / 2,
            bar_width * len(model_names) / 2,
            len(model_names),
        )

        _, ax = plt.subplots(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            normalized_values = [
                mean_stats[model_name][metric] / max_stats[metric]
                for metric in metric_names
            ]
            ax.bar(
                index + offsets[i],
                normalized_values,
                bar_width,
                label=model_name,
            )

        ax.set_ylabel("Metrics (normalized)")
        ax.set_title("Model Comparison")
        ax.set_xticks(index)
        ax.set_xticklabels(metric_names)
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(
            SparseRegression, args.config_path, args.results_dir, args.repeats
        )
    elif args.command == "plot":
        runner.plot(SparseRegression, args.config_path, args.results_dir)
    else:
        raise ValueError(f"Unknown command {args.command}.")
