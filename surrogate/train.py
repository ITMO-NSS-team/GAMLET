"""The module contains general method to train a model from `lib.lightning_modules`."""

import pickle
from typing import Dict, Any, List
from typing import Tuple

import pandas as pd
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
import torch_geometric.utils as utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from surrogate import datasets, models


def train(config: Dict[str, Any]):
    train_dataset = getattr(datasets, config["train_dataset"]["name"])(**config["train_dataset"]["parameters"])
    val_dataset = getattr(datasets, config["val_dataset"]["name"])(**config["val_dataset"]["parameters"])
    test_dataset = getattr(datasets, config["test_dataset"]["name"])(**config["test_dataset"]["parameters"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    lightning_module_class = getattr(lightning_modules, config["lightning_module"]["name"])
    lightning_module = lightning_module_class(**{k: v for k, v in config["lightning_module"].items() if k != "name"})

    logger = TensorBoardLogger(**config["tensorboard_logger"])
    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[model_checkpoint_callback, ],
    )
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lightning_module, dataloaders=test_loader)

    
def get_datasets(
        pipeline_graph_rename_path: str,
        y_pipeline_path: str,
        labels_path: str,
        pipelines_path: str,
        seed: int = 0,
) -> Tuple[List[Data], Data, Data, Data]:
    """The method makes dataset."""
    X_dataset = pd.read_csv('data/openml/X_dataset.csv').drop(columns='Unnamed: 0')
    X_dataset = X_dataset.fillna(-1)
    scaler = StandardScaler()
    X_dataset = scaler.fit_transform(X_dataset)

    with open(pipeline_graph_rename_path, 'rb') as file:
        pipeline_graph_rename = pickle.load(file)
    with open(y_pipeline_path, 'rb') as file:
        y_pipeline = list(pickle.load(file))
    with open(labels_path, 'rb') as file:
        labels = list(pickle.load(file))
    with open(pipelines_path, 'rb') as file:
        pipelines = list(pickle.load(file))

    pyg_graph = []
    p = []
    for idx, graph in enumerate(pipeline_graph_rename):
        graph = from_networkx(graph)
        graph.y = torch.tensor(int(labels[idx]), dtype=torch.float)
        graph.d = torch.tensor(X_dataset[idx], dtype=torch.float).view(1,-1)
        if graph.edge_index.size(1) != 0:
            pyg_graph.append(graph)
            p.append(pipelines[idx])

    train_dset, test_dset = train_test_split(pyg_graph, test_size=0.7, random_state=seed)
    val_dset, test_dset = train_test_split(test_dset, test_size=0.5, random_state=seed)
    return pyg_graph, train_dset, val_dset, test_dset, dict()


def train_graph_transformer(config: Dict[str, Any]):
    pyg_graph, train_dataset, val_dataset, test_dataset, meta_data = get_datasets(**config["dataset_params"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=50)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=50)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=50)

    # model_class = getattr(lightning_modules, config["lightning_module"]["name"])
    model_class = models.SurrogateModel
    
    
    # Infer parameters
    xs = []
    for dset in pyg_graph:
        for item in list(dset.x):
            xs.append(int(item))
    n_tags = len(set(xs))
    config["lightning_module"]["model_parameters"]["in_size"] = n_tags
    
    config["lightning_module"]["model_parameters"]["dim_dataset"] = dset.d.shape[1]
    
    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
        data in train_dataset])
    config["lightning_module"]["model_parameters"]["deg"] = deg
    dim_feedforward = 2 * config["lightning_module"]["model_parameters"]["d_model"]
    config["lightning_module"]["model_parameters"]["dim_feedforward"] = dim_feedforward
    config["lightning_module"]["model_parameters"]["meta_data"] = meta_data
    
    
    
    model = model_class(config["lightning_module"]["model_parameters"])
    
    logger = None#TensorBoardLogger(**config["tensorboard_logger"])
    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[model_checkpoint_callback, ],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)
