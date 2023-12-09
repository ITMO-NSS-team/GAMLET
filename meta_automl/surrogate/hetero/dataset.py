import pathlib
import pickle
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def train_collate_fn(batch):
    pipe1 = [b[0] for b in batch]
    pipe2 = [b[1] for b in batch]
    dset_data = torch.stack([b[2] for b in batch])
    y = torch.FloatTensor([b[3] for b in batch])
    return pipe1, pipe2, dset_data, y


def val_collate_fn(batch):
    task_id = torch.LongTensor([b[0] for b in batch])
    pipe_id = torch.LongTensor([b[1] for b in batch])
    pipe = [b[2] for b in batch]
    dset_data = torch.stack([b[3] for b in batch])
    y = torch.FloatTensor([b[4] for b in batch])
    return task_id, pipe_id, pipe, dset_data, y


class HeteroPipelineAndDatasetFeaturesDataset(Dataset):
    # Load from no_meta_features_and_fedot_pipelines_raw
    # The folder is made with `GAMLET/scripts/generate_dataset_with_hyperparameters_raw.py`.
    def __init__(
        self,
        task_pipe_comb_file: str,
        id2pipe: Dict[int, Union[pathlib.PosixPath, str]],
        id2dataset: Dict[int, str],
        is_val: bool = False,
    ):
        self.task_pipe_comb = pd.read_csv(task_pipe_comb_file)
        self.groups = {k: v for k, v in self.task_pipe_comb.groupby("task_id")}
        self.id2pipe = id2pipe
        self.id2dataset = id2dataset
        self.max_task_id = max(list(id2dataset.keys()))
        self.dataset_ids = list(self.groups.keys())
        self.is_val = is_val
        if self.is_val:
            self.collate_fn = val_collate_fn
        else:
            self.collate_fn = train_collate_fn

    def __len__(self):
        return len(self.task_pipe_comb)

    def _get_train_sample(self) -> Tuple[str, str, torch.Tensor, float]:
        task_id = np.random.choice(self.dataset_ids, 1).item()
        group = self.groups[task_id]
        idxes = np.random.choice(group.index, 2, replace=False)
        samples = group.loc[idxes]
        metric1, metric2 = samples.metric.to_list()
        pipe1, pipe2 = samples.pipeline_id.to_list()
        with open(self.id2pipe[pipe1], "rb") as f:
            pipe1_json_string = pickle.load(f)
        with open(self.id2pipe[pipe2], "rb") as f:
            pipe2_json_string = pickle.load(f)
        ds_data = torch.full((2,), task_id / self.max_task_id)
        if metric1 == metric2:
            label = 0.5
        elif metric1 > metric2:
            label = 1.0
        else:
            label = 1.0
        return pipe1_json_string, pipe2_json_string, ds_data, label

    def _get_val_sample(self, idx) -> Tuple[int, int, str, torch.Tensor, float]:
        sample = self.task_pipe_comb.iloc[idx]
        task_id = sample.task_id
        pipe_id = sample.pipeline_id
        with open(self.id2pipe[pipe_id], "rb") as f:
            pipe_json_string = pickle.load(f)
        metric = sample.metric
        ds_data = torch.full((2,), task_id / self.max_task_id)
        return task_id, pipe_id, pipe_json_string, ds_data, metric

    def __getitem__(self, idx):
        if self.is_val:
            return self._get_val_sample(idx)
        else:
            return self._get_train_sample()
