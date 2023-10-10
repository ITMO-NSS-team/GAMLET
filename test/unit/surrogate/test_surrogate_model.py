import numpy as np
import pandas as pd
import pytest
import yaml
import torch
import pathlib

from meta_automl.surrogate import surrogate_model

@pytest.fixture(scope='module')
def read_config():
    with open('configs/train_surrogate_model.yml') as f:
        config = yaml.load(f, yaml.Loader)
    return config

# @pytest.fixture(scope='module')
def create_model_from_config(read_config, x_pipe, x_dset):
    config = read_config
    dim_feedforward = 2 * config["model"]["model_parameters"]["d_model"]
    config["model"]["model_parameters"]["dim_feedforward"] = dim_feedforward
    config["model"]["model_parameters"]["in_size"] = x_pipe.in_size[0].item()
    config["model"]["model_parameters"]["dim_dataset"] = x_dset.x.shape[1]  
    model_class = getattr(surrogate_model, config["model"].pop("name"))
    model = model_class(**config["model"])
    return model

# def test_meta_dataset():
#     pass

def get_test_data():
    path = pathlib.Path(__file__).parent.parent.parent.resolve()
    path = path.joinpath('data/surr_data/')
    x_pipe = torch.load(path/'data_pipe_test.pt')
    x_dset = torch.load(path/'data_dset_test.pt') 
    return x_pipe, x_dset
    
def test_model_output(read_config): 
    x_pipe, x_dset = get_test_data()
    model = create_model_from_config(read_config, x_pipe, x_dset)
    pred = torch.squeeze(model.forward(x_pipe, x_dset))
    assert pred.shape[0] == 256
    
    
# def test_surrogate_advisor(read_config):   
#     config = read_config
#     advisor = SurrogateGNNPipelineAdvisor(config)
#     pipelines, scores = advisor.predict(dataset[0].get_data().x, k = 3)



