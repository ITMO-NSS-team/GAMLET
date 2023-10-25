import sys   
sys.path.append('.')  
import os
import pickle

from meta_automl.data_preparation.dataset import OpenMLDataset
from meta_automl.data_preparation.datasets_loaders import OpenMLDatasetsLoader
from meta_automl.meta_algorithm.model_advisors import SurrogateGNNPipelineAdvisor
import yaml


def main():
    # Define datasets
    dataset = OpenMLDatasetsLoader().load(['apsfailure'], allow_names=True)
    # Extract meta-features and load on demand.
    
    with open('configs/run_surrogate_model.yml') as f:
        config = yaml.load(f, yaml.Loader)
        
    with open(os.path.join(config["dataset_params"]["root_path"], "pipelines.pickle"), "rb") as input_file:
        pipelines_data = pickle.load(input_file) 
    with open(os.path.join(config["dataset_params"]["root_path"], "pipelines_fedot.pickle"), "rb") as input_file:
        pipelines_fedot = pickle.load(input_file)    

    advisor = SurrogateGNNPipelineAdvisor(config, pipelines_data, pipelines_fedot)
    
    return advisor.predict(dataset) 

if __name__ == '__main__':
    result = main()
    print(result)