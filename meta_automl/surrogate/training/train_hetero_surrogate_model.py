"""The module contains custom method to train `lib.lightning_modules.GraphTransformer`."""

import os
import pickle
from typing import Any, Dict, List, Tuple
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from meta_automl.data_preparation.surrogate_dataset.hetero.dataset import HeteroPipelineAndDatasetFeaturesDataset, HeteroPipelineDataset

from meta_automl.surrogate.hetero import surrogate_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np

torch.autograd.set_detect_anomaly(True)

def build_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:  # TODO: refactor
    # if config["model"]["model_parameters"]["dataset_encoder_type"] == "column":
    #     index_col = [0, 1]
    # else:
    #     index_col = 0
    # config["dataset_params"]["root_path"],

    print("Loading id2pipe")
    #  avoid repeated loading of mappings
    with open("/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw/id2pipeline_path.pickle", "rb") as f:
        id2pipe = pickle.load(f)
    print("Loading id2dataset")
    with open("/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw/id2dataset_id.pickle", "rb") as f:
        id2dataset = pickle.load(f)

    print("Making train dataset")
    train_dataset = HeteroPipelineDataset( #HeteroPipelineAndDatasetFeaturesDataset(
        "/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw/train_task_pipe_comb.csv",
        # "/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw/datasets.csv",
        id2pipe,
        id2dataset,
        is_val=False,
    )
    print("Making test dataset")
    val_dataset = HeteroPipelineDataset( #HeteroPipelineAndDatasetFeaturesDataset(
        "/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw/test_task_pipe_comb.csv",
        # "/home/cherniak/itmo_job/GAMLET/data/no_meta_features_and_fedot_pipelines_raw/datasets.csv",
        id2pipe,
        id2dataset,
        is_val=True,
    )
    meta_data = {
        # "dim_dataset": train_dataset.dataset_metafeatures.shape[1],
    }
    return train_dataset, val_dataset, val_dataset, meta_data

def build_model(config: Dict[str, Any], meta_data: Dict[str, Any]) -> nn.Module: # TODO: implement
    model_class = getattr(surrogate_model, config["model"]["class"])
    # config["model"]["model_parameters"]["pipeline_encoder"]["in_size"] = meta_data["in_size"]
    # config["model"]["model_parameters"]["dataset_encoder"]["dim_dataset"] = meta_data["dim_dataset"]
    dim_feedforward = 2 * config["model"]["model_parameters"]["pipeline_encoder"]["d_model"]
    config["model"]["model_parameters"]["pipeline_encoder"]["dim_feedforward"] = dim_feedforward
    model = model_class(**{k: v for k, v in config["model"].items() if k != "class"})
    return model

def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Change to 1.
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,  # Change to 1.
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=0,  # Change to 1.
        collate_fn=test_dataset.collate_fn,
    )
    return train_loader, val_loader, test_loader

def train_hetero_surrogate_model(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Create surrogate model and do training according to config parameters."""
    print("Making datasets")
    train_dataset, val_dataset, test_dataset, meta_data = build_datasets(config)
    assert len(train_dataset) != 0
    assert len(val_dataset) != 0
    assert len(test_dataset) != 0

    print("Making dataloaders")
    train_loader, val_loader, test_loader = build_dataloaders(train_dataset, val_dataset, test_dataset, config)

    print("Making model")
    model = build_model(config, meta_data)

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

    model = type(model).load_from_checkpoint(model_checkpoint_callback.best_model_path)
    print(model_checkpoint_callback.best_model_path)

    test_results = trainer.test(model, dataloaders=test_loader)
    return test_results


def test_ranking(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Test surrogate model"""
    print("Making datasets")
    _, _, test_dataset, _ = build_datasets(config)
    assert len(test_dataset) != 0

    print("Making dataloaders")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=0,  # Change to 1.
        collate_fn=test_dataset.collate_fn,
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
