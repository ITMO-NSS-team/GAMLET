"""The module contains custom method to train `lib.lightning_modules.GraphTransformer`."""

import json
import os
import pickle
import random
import warnings
from typing import Any, Dict, List, Tuple, Union


import time

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from fedot.core.pipelines.pipeline import Pipeline

from meta_automl.surrogate import models
from meta_automl.surrogate.datasets import GraphDataset, PairDataset, SingleDataset
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor

 
def get_datasets(path, is_pair = False, binary_y = True):
    """Loading preprocessed data and creating Dataset objects for model training 
    Parameters:
    -----------
    is_pair: create dataset with or without pipeline pairs.
    
    """      
    with open(os.path.join(path, "pipelines.pickle"), "rb") as input_file:
        pipelines = pickle.load(input_file)   
    
    for p in pipelines
    
    datasets = pd.read_csv(os.path.join(path, 'datasets.csv'), index_col=0).fillna(0)
    task_pipe_comb = pd.read_csv(os.path.join(path, 'task_pipe_comb.csv'))
    
    # task_pipe_comb_bin = task_pipe_comb.sort_values(by = 'y', ascending = False)
    # task_pipe_comb_bin = task_pipe_comb_bin.groupby('task_id').apply(to_labels)
    # task_pipe_comb_bin = task_pipe_comb_bin.groupby('task_id').apply(lambda x: x)

    VAL_R = 0.15
    TEST_R = 0.15
    tasks_in_file = set(datasets.index)
    try:
        with open(os.path.join(path, "split.json")) as f:
            splits = json.load(f)
        train_task_set, val_task_set, test_task_set = train_val_test_split(splits)
        train_task_set = train_task_set & tasks_in_file
        val_task_set = val_task_set & tasks_in_file
        test_task_set = test_task_set & tasks_in_file     
    except FileNotFoundError:
        train_task_set, val_task_set, test_task_set = \
            random_train_val_test_split(list(tasks_in_file), 
                                        (VAL_R,TEST_R) )
    if len(val_task_set) == 0:
        train_task_set, val_task_set = random_train_val_test_split(list(train_task_set), 
                                        (VAL_R,) )
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
            task_pipe_comb[task_pipe_comb.task_id.isin(val_task_set)].reset_index(drop=True),
            GraphDataset(pipelines),
            datasets,
        )
    test_dataset = SingleDataset(
            task_pipe_comb[task_pipe_comb.task_id.isin(test_task_set)].reset_index(drop=True),
            GraphDataset(pipelines),
            datasets,
        )   

    # Infer parameters
    meta_data = dict()
    # xs = []
    # for dset in pipelines:
    #     for item in list(dset.x):
    #         xs.append(int(item))
    # n_tags = len(set(xs))
    meta_data["in_size"] = len(pipelines[0].in_size)
    meta_data["dim_dataset"] = datasets.shape[1]
    return train_dataset, val_dataset, test_dataset, meta_data

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
    return set(train_task_set), set(val_task_set), set(test_task_set)

def random_train_val_test_split(tasks: List[int], splits: List[float]) -> Tuple[List[int], List[int], List[int]]:
    """Split tasks list into train/valid/test sets randomly"""
    random.seed(10)                                                              
    random.shuffle(tasks)
    ind_splits = [len(tasks)]
    split_ratio = 0
    for split in reversed(splits):
        split_ratio += split
        ind_splits.append(int((1- split_ratio)*len(tasks)) )
    
    task_sets = []
    ind_prev = 0
    for ind in reversed(ind_splits):
        task_sets.append(set(tasks[ind_prev:ind]) )
        ind_prev = ind
    return task_sets

def train_surrogate_model(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Create surrogate model and do training according to config parameters"""    
    is_pair = False
    model_class = getattr(models, config["model"].pop("name"))
    if model_class.__name__ == 'RankingPipelineDatasetSurrogateModel':
        is_pair = True

    train_dataset,  val_dataset, test_dataset, meta_data = get_datasets(
        config["dataset_params"]["root_path"], is_pair)
    
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


def test_ranking(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Test surrogate model"""    

    train_dataset,  val_dataset, test_dataset, meta_data = get_datasets(
        config["dataset_params"]["root_path"], False, False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=config["num_dataloader_workers"],
    )

    model_class = getattr(models, config["model"].pop("name"))    
    surrogate_model = model_class.load_from_checkpoint(
            checkpoint_path=config["model_data"]["save_dir"]+"checkpoints/last.ckpt",
            hparams_file=config["model_data"]["save_dir"]+"hparams.yaml"
        )
    surrogate_model.eval()
    
    task_ids, pipe_ids, y_preds, y_trues = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            task_id, pipe_id, x_graph, x_dset, y_true = batch
            y_pred = surrogate_model(x_graph, x_dset)
            y_pred = torch.squeeze(y_pred)
            task_ids.append(task_id)
            pipe_ids.append(pipe_id)
            y_preds.append(y_pred)
            y_trues.append(y_true)
            
    df = pd.DataFrame({'task_id': np.concatenate(task_ids),
                           'pipe_id': np.concatenate(pipe_ids),
                           'y_pred': np.concatenate(y_preds),
                           'y_true': np.concatenate(y_trues)})
    df.to_csv('results.csv')

