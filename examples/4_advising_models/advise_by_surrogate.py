import sys   
sys.path.append('.')  
import pandas as pd

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.meta_algorithm.model_advisors import SurrogateGNNPipelineAdvisor

import yaml


def main():
    # Define datasets.
    dataset = OpenMLDatasetsLoader().load(['apsfailure'], allow_names=True)
    # Extract meta-features and load on demand.
    
    with open('configs/run_surrogate_model.yml') as f:
        config = yaml.load(f, yaml.Loader)
    advisor = SurrogateGNNPipelineAdvisor(config)
    
    pipelines, scores = advisor.predict(dataset[0].get_data().x, k = 3)
    return pipelines


if __name__ == '__main__':
    result = main()
    print(result)