import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from numpy.typing import ArrayLike
from sklearn.linear_model import (
    OrthogonalMatchingPursuit,
    Lasso,
    ElasticNet,
    ElasticNetCV,
)
from sklearn.metrics import mean_squared_error, f1_score

from expflow import Experiment, Runner


class RegularizationPath(ABC):

    @abstractmethod
    def fit_path(
        self,
        A: ArrayLike,
        y: ArrayLike,
        max_nnz: int = 10,
        max_solve_time: float = 60.0,
        **kwargs,
    ) -> dict: ...


class LassoPath(RegularizationPath):

    def fit_path(
        self,
        A: ArrayLike,
        y: ArrayLike,
        max_nnz: int = 10,
        max_solve_time: float = 60.0,
        alpha_ratio: float = 1e-4,
        alpha_num: int = 40,
        fit_intercept: bool = False,
    ) -> list[dict]:

        alpha_max = np.linalg.norm(A.T @ y, np.inf) / y.size
        alpha_min = alpha_ratio * alpha_max
        alpha_grid = np.logspace(
            np.log10(alpha_max),
            np.log10(alpha_min),
            alpha_num,
        )

        path = []
        for alpha in alpha_grid:

            solver = Lasso(alpha=alpha, fit_intercept=fit_intercept)

            start_time = time.time()
            x = solver.fit(A, y).coef_.flatten()
            solve_time = time.time() - start_time

            if len(np.flatnonzero(x)) > max_nnz:
                break
            if solve_time > max_solve_time:
                break

            path.append({"x": x, "solve_time": solve_time})

        return path


class EnetPath(RegularizationPath):

    def fit_path(
        self,
        A: ArrayLike,
        y: ArrayLike,
        max_nnz: int = 10,
        max_solve_time: float = 60.0,
        alpha_ratio: float = 1e-4,
        alpha_num: int = 40,
        fit_intercept: bool = False,
    ) -> list[dict]:

        enet_cv = ElasticNetCV()
        enet_cv.fit(A, y)
        l1_ratio = enet_cv.l1_ratio_

        alpha_max = np.linalg.norm(A.T @ y, np.inf) / y.size
        alpha_min = alpha_ratio * alpha_max
        alpha_grid = (
            np.logspace(
                np.log10(alpha_max),
                np.log10(alpha_min),
                alpha_num,
            )
            / l1_ratio
        )

        path = []
        for alpha in alpha_grid:

            solver = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept
            )

            start_time = time.time()
            x = solver.fit(A, y).coef_.flatten()
            solve_time = time.time() - start_time

            if len(np.flatnonzero(x)) > max_nnz:
                break
            if solve_time > max_solve_time:
                break

            path.append({"x": x, "solve_time": solve_time})

        return path


class OmpPath(RegularizationPath):

    def fit_path(
        self,
        A: ArrayLike,
        y: ArrayLike,
        max_nnz: int = 10,
        max_solve_time: float = 60.0,
        fit_intercept: bool = False,
    ) -> list[dict]:

        path = []

        for nnz in range(1, max_nnz + 1):

            solver = OrthogonalMatchingPursuit(
                n_nonzero_coefs=nnz,
                fit_intercept=fit_intercept,
            )

            start_time = time.time()
            x = solver.fit(A, y).coef_.flatten()
            solve_time = time.time() - start_time

            if len(np.flatnonzero(x)) > max_nnz:
                break
            if solve_time > max_solve_time:
                break

            path.append({"x": x, "solve_time": solve_time})

        return path


class SparseRegression(Experiment):

    def generate_data(
        self,
        t: float = 0.0,
        k: int = 10,
        m: int = 100,
        n: int = 300,
        r: float = 0.9,
        s: float = 10.0,
    ) -> List[ArrayLike]:
        x = np.zeros(n)
        S = np.random.choice(n, k, replace=False)
        if t == 0:
            x[S] = np.sign(np.random.randn(k))
        else:
            x[S] = np.random.normal(0.0, t, k)
            x[S] += np.sign(x[S])
        M = np.zeros(n)
        K1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
        K2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
        K = np.power(r, np.abs(K1 - K2))
        A = np.random.multivariate_normal(M, K, size=m)
        A /= np.linalg.norm(A, axis=0)
        y = A @ x
        e = np.random.randn(m)
        e *= np.linalg.norm(y) / (s * np.linalg.norm(e))
        y += e
        return A, y, x

    def setup(self) -> None:
        self.A, self.y, self.x_true = self.generate_data(
            **self.config["dataset"]
        )

    def get_model(self, model_name: str) -> RegularizationPath:
        bindings = {
            "lasso": LassoPath(),
            "enet": EnetPath(),
            "omp": OmpPath(),
        }
        return bindings[model_name]

    def get_stat(self, stat_name: str, path_item: dict) -> float:
        if stat_name == "mse":
            return mean_squared_error(self.y, self.A @ path_item["x"])
        elif stat_name == "f1s":
            if self.x_true is not None:
                return f1_score(self.x_true != 0.0, path_item["x"] != 0.0)
            else:
                return 0.0
        elif stat_name == "time":
            return path_item["solve_time"]
        raise ValueError("Unknown stat name")

    def run(self) -> dict:
        result = {}
        for model_name, model_params in self.config["models"].items():
            print(f"Running {model_name}...")

            model = self.get_model(model_name)
            params = {**model_params, **self.config["common_params"]}
            path = model.fit_path(self.A, self.y, **params)

            stats = {
                nnz: {stat_name: np.nan for stat_name in self.config["stats"]}
                for nnz in range(self.config["common_params"]["max_nnz"] + 1)
            }
            for path_item in path:
                nnz = np.count_nonzero(path_item["x"])
                for stat_name in self.config["stats"]:
                    stats[nnz][stat_name] = self.get_stat(stat_name, path_item)

            result[model_name] = stats

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list[dict]) -> None:

        grid_nnz = range(self.config["common_params"]["max_nnz"] + 1)

        stats = {
            stat_name: {
                model_name: [[] for _ in grid_nnz]
                for model_name in self.config["models"]
            }
            for stat_name in self.config["stats"]
        }

        for result in results:
            for model_name, path_stats in result.items():
                for nnz in grid_nnz:
                    for stat_name in self.config["stats"]:
                        stats[stat_name][model_name][nnz].append(
                            path_stats[nnz][stat_name]
                        )

        _, axs = plt.subplots(1, len(stats), figsize=(10, 6))

        for i, (stat_name, stat_values) in enumerate(stats.items()):
            for model_name, model_stats in stat_values.items():
                y_values = [
                    (
                        np.nanmean(model_stats[nnz])
                        if not np.all(np.isnan(model_stats[nnz]))
                        else np.nan
                    )
                    for nnz in grid_nnz
                ]
                axs[i].plot(
                    grid_nnz,
                    y_values,
                    marker=".",
                    label=model_name,
                ),
            axs[i].set_xlabel("nnz")
            axs[i].set_ylabel(stat_name)
            axs[i].set_yscale(self.config["stats"][stat_name]["scale"])
        axs[0].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--result_dir", type=Path)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(
            SparseRegression, args.config_path, args.result_dir, args.repeats
        )
    elif args.command == "plot":
        runner.plot(
            SparseRegression,
            args.config_path,
            args.result_dir,
            args.save_dir,
        )
    else:
        raise ValueError(f"Unknown command {args.command}.")
