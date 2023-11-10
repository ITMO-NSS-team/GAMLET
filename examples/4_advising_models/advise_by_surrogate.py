import os
import pickle
from pathlib import Path

import yaml

from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.data_preparation.file_system.file_system import get_configs_dir, get_project_root
from meta_automl.meta_algorithm.model_advisors import SurrogateGNNPipelineAdvisor


def main():
    # Define datasets
    dataset = OpenMLDatasetsLoader().load(["apsfailure"], allow_names=True)
    # Extract meta-features and load on demand.
    with open(get_configs_dir() / 'run_surrogate_model.yml') as f:
        config = yaml.load(f, yaml.Loader)

    project_root = get_project_root()
    pipelines_data_path = Path(config["dataset_params"]["root_path"], "pipelines.pickle")
    pipelines_fedot_data_path = os.path.join(config["dataset_params"]["root_path"], "pipelines_fedot.pickle")
    pipelines_data_path = project_root / pipelines_data_path
    pipelines_fedot_data_path = project_root / pipelines_fedot_data_path
    with open(pipelines_data_path, "rb") as input_file:
        pipelines_data = pickle.load(input_file)
    with open(pipelines_fedot_data_path, "rb") as input_file:
        pipelines_fedot = pickle.load(input_file)

    advisor = SurrogateGNNPipelineAdvisor(config, pipelines_data, pipelines_fedot)

    return advisor.predict(dataset)


if __name__ == "__main__":
    result = main()
    print(result)
