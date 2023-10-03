"""The module contains custom method to train `lib.lightning_modules.GraphTransformer`."""

import json
import os
import pickle
import random
import warnings
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from surrogate import models
from surrogate.datasets import GraphDataset, PairDataset, SingleDataset


def get_datasets(path, is_pair = False):
    with open(os.path.join(path, "pipelines.pickle"), "rb") as input_file:
        pipelines = pickle.load(input_file)

    task_pipe_comb = pd.read_csv(os.path.join(path, 'task_pipe_comb.csv'), index_col=0)
    datasets = np.genfromtxt(os.path.join(path, 'datasets.csv'), delimiter=",")

    K_TOP = 3
    to_labels = partial(to_labels_k, klim=K_TOP)
    task_pipe_comb_bin = task_pipe_comb.sort_values(by = 'y', ascending = False)
    task_pipe_comb_bin = task_pipe_comb_bin.groupby('task_id').apply(to_labels)

    try:
        with open(os.path.join(path, "split.json")) as f:
            splits = json.load(f)
        train_task_set, val_task_set, test_task_set = train_val_test_split(splits)
    except FileNotFoundError:
        train_task_set, val_task_set, test_task_set = random_train_val_test_split(datasets)

    if is_pair:
        train_dataset = PairDataset(
        task_pipe_comb[task_pipe_comb.task_id.isin(train_task_set)].reset_index(drop=True),
        GraphDataset(pipelines),
        datasets,
        )
    else:
        train_dataset = SingleDataset(
            task_pipe_comb[task_pipe_comb.task_id.isin(train_task_set)].reset_index(drop=True),
            GraphDataset(pipelines),
            datasets,
        )
    val_dataset = SingleDataset(
        task_pipe_comb_bin[task_pipe_comb_bin.task_id.isin(val_task_set)].reset_index(drop=True),
        GraphDataset(pipelines),
        datasets,
    )
    test_dataset = SingleDataset(
        task_pipe_comb_bin[task_pipe_comb_bin.task_id.isin(test_task_set)].reset_index(drop=True),
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

def train_val_test_split(splits: Dict[str, List[int]]) -> Tuple[List[int], List[int], List[int]]:
    try:
        train_task_set = splits["train"]
    except KeyError:
        raise KeyError("The key `train` is not found in the json file.")
    try:
        val_task_set = splits["val"]
    except KeyError:
        warnings.warn("The key `val` is not found in the json file.")
        val_task_set = []
    try:
        test_task_set = splits["test"]
    except KeyError:
        warnings.warn("The key `test` is not found in the json file.")
        test_task_set = []
    return train_task_set, val_task_set, test_task_set

def random_train_val_test_split(datasets: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
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
    is_pair = False
    model_class = getattr(models, config["model"].pop("name"))
    if model_class.__name__ == 'RankingSurrogateModel':
        is_pair = True

    train_dataset,  val_dataset, test_dataset, meta_data = get_datasets(
        config["dataset_params"]["root_path"], is_pair)

    # Not the best solution, that may lead to overfitting, but enables fair comparison with traditional models.
    if len(val_dataset) == 0:
        config["early_stopping_callback"]["monitor"] = "train_loss"
        config["model_checkpoint_callback"]["monitor"] = "train_loss"

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

    checkpoint = torch.load(model_checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # model = model_class.load_from_checkpoint(model_checkpoint_callback.best_model_path)

    test_results = trainer.test(model, dataloaders=test_loader)
    return test_results
