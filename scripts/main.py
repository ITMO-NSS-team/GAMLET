import argparse
import sys

import yaml

# sys.path.append("..")
from surrogate import training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--tune", action="store_true", default=False)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)
    if args.train and args.tune:
        raise ValueError("Cannot train and tune simultaneously")
    elif args.train:
        training_method = getattr(training, config["training_method"])
        training_method(config)
    elif args.tune:
        tuning_method = getattr(training, config["tuning_method"])
        n_trials = config.pop("n_trials")
        tuning_method(config, n_trials=n_trials)
