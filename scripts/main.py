import argparse
import sys

import yaml

sys.path.append("..")
from surrogate import training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)
    training_method = getattr(training, config["training_method"])
    training_method(config)
