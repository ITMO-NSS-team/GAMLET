import pytest
import torch
import yaml

from gamlet.data_preparation.file_system.file_system import get_configs_dir, get_data_dir
from gamlet.surrogate import surrogate_model


@pytest.fixture(scope='module')
def read_config():
    with open(get_configs_dir() / 'train_surrogate_model.yml') as f:
        config = yaml.load(f, yaml.Loader)
    return config


def create_model_from_config(read_config, x_pipe, x_dset):
    config = read_config
    dim_feedforward = 2 * config["model"]["model_parameters"]["d_model"]
    config["model"]["model_parameters"]["dim_feedforward"] = dim_feedforward
    config["model"]["model_parameters"]["in_size"] = x_pipe.in_size[0].item()
    config["model"]["model_parameters"]["dim_dataset"] = x_dset.x.shape[1]
    model_class = getattr(surrogate_model, config["model"].pop("name"))
    model = model_class(**config["model"])
    return model


def get_test_data():
    path = get_data_dir() / 'surr_data'
    x_pipe = torch.load(path / 'data_pipe_test.pt')
    x_dset = torch.load(path / 'data_dset_test.pt')
    return x_pipe, x_dset


def test_model_output(read_config):
    x_pipe, x_dset = get_test_data()
    model = create_model_from_config(read_config, x_pipe, x_dset)
    pred = torch.squeeze(model.forward(x_pipe, x_dset))
    assert pred.shape[0] == 256
