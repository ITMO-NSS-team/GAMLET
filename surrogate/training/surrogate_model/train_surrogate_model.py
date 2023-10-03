"""The module contains custom method to train `lib.lightning_modules.GraphTransformer`."""

import pickle
from typing import Dict, Any, List

import numpy as np

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx

from surrogate import models

from surrogate.datasets import SingleDataset, GraphDataset, PairDataset

from functools import partial
import random

def preprocess_raw_files(path):
    X_dataset = pd.read_csv(path + 'X_dataset.csv').drop(columns='Unnamed: 0')
    X_task_id = pd.read_csv(path + 'X_task_id.csv').drop(columns='Unnamed: 0')

    X_dataset = X_dataset.fillna(-1)
    scaler = StandardScaler()
    X_dataset = scaler.fit_transform(X_dataset)

    with open(path + 'pipelines_graphs/pipeline_graph_rename.pickle', 'rb') as file:
        pipeline_graph_rename = pickle.load(file)
    with open(path + 'pipelines_graphs/y.pickle', 'rb') as file:
        y_pipeline = list(pickle.load(file))
    with open(path + 'pipelines_graphs/labels.pickle', 'rb') as file:
        labels = list(pickle.load(file))
    with open(path + '/pipelines_graphs/pipelines.pickle', 'rb') as file:
        pipelines = list(pickle.load(file))

    uniq_pipelines = []
    pipeline_ids= []
    pipeline_map = dict()
    ind = 0
    for i,p in enumerate(pipelines):
        if p not in pipeline_map:
            pyg_data = from_networkx(pipeline_graph_rename[i])

            if pyg_data.edge_index.size(1) != 0:
                pipeline_map[p] = ind
                uniq_pipelines.append(pyg_data)
                ind += 1
            else:
                pipeline_map[p] = None
        pipeline_ids.append(pipeline_map[p])

    d_codes = X_task_id.task_id.astype("category").cat.codes
    dict_tasks = dict(  zip(d_codes.values, np.arange(len(d_codes)))  )
    x_dataset = X_dataset[[dict_tasks[i] for i in range(len(dict_tasks))]]

    X_task_id['pipeline_id'] = pipeline_ids
    X_task_id['y'] = y_pipeline
    X_task_id['task_id'] = d_codes

    X_task_id = X_task_id.dropna()
    X_task_id['pipeline_id'] = X_task_id['pipeline_id'].astype(int)

    labels = []
    for p in uniq_pipelines:
        labels.append(p.x.numpy())
    le = LabelEncoder()
    le.fit(np.concatenate(labels))
    for p in uniq_pipelines:
        p.x = torch.tensor(le.transform(p.x.numpy()))

    X_task_id.to_csv('task_pipe_comb.csv')
    np.savetxt("datasets.csv", x_dataset, delimiter=",")
    with open('pipelines.pickle', 'wb') as f:
        pickle.dump(uniq_pipelines, f)


def get_datasets(path):
    with open(path + "pipelines.pickle", "rb") as input_file:
        pipelines = pickle.load(input_file)

    task_pipe_comb = pd.read_csv(path +'task_pipe_comb.csv', index_col=0)
    datasets = np.genfromtxt(path +'datasets.csv', delimiter=",")

    K_TOP = 3
    to_labels = partial(to_labels_k, klim=K_TOP)
    task_pipe_comb_bin = task_pipe_comb.sort_values(by = 'y', ascending = False)
    task_pipe_comb_bin = task_pipe_comb_bin.groupby('task_id').apply(to_labels)

    train_task_set, val_task_set, test_task_set = train_test_split(datasets)


    train_dataset = SingleDataset(
        task_pipe_comb[task_pipe_comb.task_id.isin(train_task_set)],
        GraphDataset(pipelines),
        datasets,
    )
    val_dataset = SingleDataset(
        task_pipe_comb_bin[task_pipe_comb_bin.task_id.isin(val_task_set)],
        GraphDataset(pipelines),
        datasets,
    )
    test_dataset = SingleDataset(
        task_pipe_comb_bin[task_pipe_comb_bin.task_id.isin(test_task_set)],
        GraphDataset(pipelines),
        datasets,
    )
    # Infer parameters
    meta_data = dict()
    xs = []
    for dset in pipelines:
        for item in list(dset.x):
            xs.append(int(item))
    n_tags = len(set(xs))
    meta_data["in_size"] = n_tags
    meta_data["dim_dataset"] = datasets.shape[1]
    return train_dataset, val_dataset, test_dataset, meta_data

def to_labels_k(x, klim):
    vals = np.zeros(len(x))
    vals[:klim] = 1
    x['y'] = vals
    return x

def train_test_split(datasets):
    random.seed(10)
    tasks = list(range(len(datasets)))
    VAL_R = 0.15
    TEST_R = 0.15
    random.shuffle(tasks)

    train_ind = int((1- (VAL_R+TEST_R))*len(datasets))
    val_ind = train_ind + int(VAL_R*len(datasets))

    train_task_set = set(tasks[:train_ind])
    val_task_set = set(tasks[train_ind:val_ind])
    test_task_set = set(tasks[val_ind:])
    return train_task_set, val_task_set, test_task_set


def train_surrogate_model(config: Dict[str, Any]) -> List[Dict[str, float]]:
    train_dataset,  val_dataset, test_dataset, meta_data = get_datasets('data/openml/')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_dataloader_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )

    model_class = getattr(models, config["model"].pop("name"))

    config["model"]["model_parameters"]["in_size"] = meta_data["in_size"]
    config["model"]["model_parameters"]["dim_dataset"] = meta_data["dim_dataset"]
    dim_feedforward = 2 * config["model"]["model_parameters"]["d_model"]
    config["model"]["model_parameters"]["dim_feedforward"] = dim_feedforward
    config["model"]["model_parameters"]["meta_data"] = {}
    model = model_class(**config["model"])

    if config["tensorboard_logger"] is not None:
        logger = TensorBoardLogger(**config["tensorboard_logger"])
    else:
        logger = None

    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    if config["early_stopping_callback"] is not None:
        early_stopping_callback = EarlyStopping(**config["early_stopping_callback"])
    else:
        early_stopping_callback = None

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[c for c in [model_checkpoint_callback, early_stopping_callback] if c is not None],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = trainer.test(model, dataloaders=test_loader)
    return test_results
