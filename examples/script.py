import argparse
import pathlib


def create_experiment(expname):

    exp_dir = pathlib.Path(__file__).parent.joinpath(expname)
    exp_dir.mkdir()

    exp_results_dir = exp_dir.joinpath("results")
    exp_results_dir.mkdir()

    exp_run_file = exp_dir.joinpath("run.py")
    exp_run_file.touch()
    with open(exp_run_file, "w") as file:
        file.write("from experiment import Experiment\n\n")
        file.write("# Write your experiment here\n")

    exp_config_file = exp_dir.joinpath("config.yaml")
    exp_config_file.touch()
    with open(exp_config_file, "w") as file:
        file.write("# Write your configuration here\n")

    exp_gitkeep_file = exp_results_dir.joinpath(".gitkeep")
    exp_gitkeep_file.touch()


parser = argparse.ArgumentParser()
parser.add_argument("command", type=str, choices=["create"])
parser.add_argument("expname", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "create":
        print(f"Creating experiment {args.expname}")
        create_experiment(args.expname)
    else:
        raise ValueError(f"Unknown command {args.command}")
