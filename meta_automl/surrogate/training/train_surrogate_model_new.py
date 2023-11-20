"""The module contains custom method to train `lib.lightning_modules.GraphTransformer`."""

import json
import os
import pickle
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch


from meta_automl.surrogate import surrogate_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np

class NewDataset(Dataset):
    def __init__(self, task_pipe_comb_file: str, id2pipe, id2dataset, is_val=False):
        self.task_pipe_comb = pd.read_csv(task_pipe_comb_file)
        self.groups = {k: v for k, v in self.task_pipe_comb.groupby('task_id')}  # Is not required if `is_val` == `True`.
        self.id2pipe = id2pipe
        self.id2dataset = id2dataset
        self.max_task_id = max(list(id2dataset.keys()))
        self.dataset_ids = list(self.groups.keys())
        self.is_val = is_val
            
    def __len__(self):
        return len(self.task_pipe_comb)
    
    def _get_train_sample(self):
        task_id = np.random.choice(self.dataset_ids, 1).item()
        group = self.groups[task_id]
        idxes = np.random.choice(group.index, 2, replace=False)
        samples = group.loc[idxes]
        metric1, metric2 = samples.metric.to_list()
        pipe1, pipe2 = samples.pipeline_id.to_list()
        with open(self.id2pipe[pipe1], "rb") as f:
            pipe1 = pickle.load(f)
        with open(self.id2pipe[pipe2], "rb") as f:
            pipe2 = pickle.load(f)
        ds_data = torch.full((2,), task_id / self.max_task_id)
        if metric1 == metric2:
            label = 0.5
        elif metric1 > metric2:
            label = 1.0
        else:
            label = 1.0
        return pipe1, pipe2, ds_data, label
    
    def _get_val_sample(self, idx):
        sample = self.task_pipe_comb.iloc[idx]
        task_id = sample.task_id
        pipe_id = sample.pipeline_id
        with open(self.id2pipe[pipe_id], "rb") as f:
            pipe = pickle.load(f)
        metric = sample.metric
        ds_data = torch.full((2,), task_id / self.max_task_id)
        return task_id, pipe_id, pipe, ds_data, metric
    
    def __getitem__(self, idx):
        if self.is_val:
            return self._get_val_sample(idx)
        else:
            return self._get_train_sample()
            
        
        

def get_datasets(path, is_pair=False, index_col=0):
    print("Loading id2pipe")
    with open("/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/id2pipeline_path.pickle", "rb") as f:
            id2pipe = pickle.load(f)
    print("Loading id2dataset")
    with open("/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/id2dataset_id.pickle", "rb") as f:
        id2dataset = pickle.load(f)    
    
    print("Making train dataset")
    train_dataset = NewDataset(
        "/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/train_task_pipe_comb.csv",
        id2pipe,
        id2dataset,
        is_val=False,
    )
    print("Making test dataset")
    test_dataset = NewDataset(
        "/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/test_task_pipe_comb.csv",
        id2pipe,
        id2dataset,
        is_val=True,
    )
    meta_data = {
        "in_size": 4,  # TODO: change
        "dim_dataset": 2,
    }
    return train_dataset, test_dataset, test_dataset, meta_data

def train_collate_fn(batch):
    pipe1 = Batch.from_data_list([b[0] for b in batch])
    pipe2 = Batch.from_data_list([b[1] for b in batch])
    dset_data = torch.stack([b[2] for b in batch])
    y = torch.FloatTensor([b[3] for b in batch])
    return pipe1, pipe2, dset_data, y

def val_collate_fn(batch):
    task_id = torch.LongTensor([b[0] for b in batch])
    pipe_id = torch.LongTensor([b[1] for b in batch])
    pipe = Batch.from_data_list([b[2] for b in batch])
    dset_data = torch.stack([b[3] for b in batch])
    y = torch.FloatTensor([b[4] for b in batch])
    return task_id, pipe_id, pipe, dset_data, y

def train_surrogate_model_new(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Create surrogate model and do training according to config parameters."""
    is_pair = False
    model_class = getattr(surrogate_model, config["model"].pop("name"))
    if model_class.__name__ == 'RankingPipelineDatasetSurrogateModel':
        is_pair = True

    if config["model"]["model_parameters"]["dataset_encoder_type"] == "column":
        index_col = [0, 1]
    else:
        index_col = 0

    print("Making datasets")
    train_dataset, val_dataset, test_dataset, meta_data = get_datasets(config["dataset_params"]["root_path"], is_pair,
                                                                       index_col=index_col)
    assert len(train_dataset) != 0
    assert len(val_dataset) != 0
    assert len(test_dataset) != 0
    print("Making dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        collate_fn=val_collate_fn,
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        collate_fn=val_collate_fn,
    )

    print("Making model")
    config["model"]["model_parameters"]["in_size"] = meta_data["in_size"]
    config["model"]["model_parameters"]["dim_dataset"] = meta_data["dim_dataset"]
    dim_feedforward = 2 * config["model"]["model_parameters"]["d_model"]
    config["model"]["model_parameters"]["dim_feedforward"] = dim_feedforward
    model = model_class(**config["model"])

    print("Making auxiliary stuff")
    if config["tensorboard_logger"] is not None:
        logger = TensorBoardLogger(**config["tensorboard_logger"])
    else:
        logger = None

    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    if config["early_stopping_callback"] is not None:
        early_stopping_callback = EarlyStopping(**config["early_stopping_callback"])
    else:
        early_stopping_callback = None

    print("Making trainer")
    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[c for c in [model_checkpoint_callback, early_stopping_callback] if c is not None],
        gradient_clip_val=0.5,
    )
    print("Training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model = model_class.load_from_checkpoint(model_checkpoint_callback.best_model_path)
    print(model_checkpoint_callback.best_model_path)

    test_results = trainer.test(model, dataloaders=test_loader)
    return test_results


def test_ranking(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Test surrogate model"""
    if config["model"]["model_parameters"]["dataset_encoder_type"] == "column":
        index_col = [0, 1]
    else:
        index_col = 0

    _, _, test_dataset, _ = get_datasets(config["dataset_params"]["root_path"], False, index_col=index_col)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=config["num_dataloader_workers"],
    )
    model_class = getattr(surrogate_model, config["model"].pop("name"))
    chpoint_dir = config["model_data"]["save_dir"] + "checkpoints/"
    model = model_class.load_from_checkpoint(
        checkpoint_path=chpoint_dir + os.listdir(chpoint_dir)[0],
        hparams_file=config["model_data"]["save_dir"] + "hparams.yaml"
    )
    model.eval()

    task_ids, pipe_ids, y_preds, y_trues = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            surrogate_model.test_step(batch)
            res = surrogate_model.test_step_outputs.pop()
            task_ids.append(res['task_id'])
            pipe_ids.append(res['pipe_id'])
            y_preds.append(res['y_pred'])
            y_trues.append(res['y_true'])

    df = pd.DataFrame({'task_id': np.concatenate(task_ids),
                       'pipe_id': np.concatenate(pipe_ids),
                       'y_pred': np.concatenate(y_preds),
                       'y_true': np.concatenate(y_trues)})

    with open(config["dataset_params"]["root_path"] + "/pipelines_fedot.pickle", "rb") as input_file:
        pipelines_fedot = pickle.load(input_file)

    res = df.loc[df.groupby(['task_id'])['y_pred'].idxmax()]
    res['model_str'] = [str(pipelines_fedot[i]) for i in res.pipe_id.values]
    res = res[['task_id', 'y_true', 'model_str']]
    res['y_true'] = -res['y_true']
    res.columns = ['dataset', 'fitness', 'model_str']

    res.to_csv('surrogate_test_set_prediction.csv', index=False)


if __name__ == "__main__":
    model_class = surrogate_model.RankingPipelineDatasetSurrogateModel
    # model_class = getattr(surrogate_model, config["model"].pop("name"))
    # config["model"]["model_parameters"]["in_size"] = meta_data["in_size"]
    # config["model"]["model_parameters"]["dim_dataset"] = meta_data["dim_dataset"]
    # dim_feedforward = 2 * config["model"]["model_parameters"]["d_model"]
    # config["model"]["model_parameters"]["dim_feedforward"] = dim_feedforward
    # model = model_class(**config["model"])
    
    ckpt_file = "/Users/cherniak/itmo_job/GAMLET/experiment_logs/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/version_3/checkpoints/epoch=0-step=102.ckpt"
    model = model_class.load_from_checkpoint(ckpt_file)
    
    with open("/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/id2pipeline_path.pickle", "rb") as f:
        id2pipe = pickle.load(f)
    with open("/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/id2dataset_id.pickle", "rb") as f:
        id2dataset = pickle.load(f)
    
    test_dataset = NewDataset(
        "/Users/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines__no_type_with_hyperparams(4_with_learnables)/test_task_pipe_comb.csv",
        id2pipe,
        id2dataset,
        is_val=True,
    )
    meta_data = {
        "in_size": 4,  # TODO: change
        "dim_dataset": 2,
    }
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8192, #config["batch_size"],
        num_workers=0,
        collate_fn=val_collate_fn,
    )
    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator="mps",
        devices="auto",
        gradient_clip_val=0.5,
    )
    
    test_results = trainer.test(model, dataloaders=test_loader)