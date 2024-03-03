import os
import pathlib
import pickle
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from meta_automl.data_preparation.surrogate_dataset.hetero.data_types import HeterogeneousBatch, HeterogeneousData
from meta_automl.data_preparation.pipeline_features_extractors import fedot_pipeline_features_extractor


class HeteroPipelineAndDatasetFeaturesDataset(Dataset):
    # Load from no_meta_features_and_fedot_pipelines_raw
    # The folder is made with `GAMLET/scripts/generate_dataset_with_hyperparameters_raw.py`.
    def __init__(
        self,
        task_pipe_comb_file: str,
        dataset_metafeatures_path: str,
        id2pipe: Dict[int, Union[pathlib.PosixPath, str]],
        id2dataset: Dict[int, str],
        is_val: bool = False,
    ):
        self.task_pipe_comb = pd.read_csv(task_pipe_comb_file)
        self.dataset_metafeatures = pd.read_csv(dataset_metafeatures_path)
        self.groups = {k: v for k, v in self.task_pipe_comb.groupby("task_id")}
        self.id2pipe = id2pipe
        self.id2dataset = id2dataset
        self.dataset_ids = list(self.groups.keys())
        self.is_val = is_val
        if self.is_val:
            self.collate_fn = self.val_collate_fn
        else:
            self.collate_fn = self.train_collate_fn

    @staticmethod
    def train_collate_fn(batch) -> Tuple[HeterogeneousBatch, HeterogeneousBatch, Batch, torch.Tensor]:
        pipe1 = HeterogeneousBatch.from_heterogeneous_data_list([b[0] for b in batch])
        pipe2 = HeterogeneousBatch.from_heterogeneous_data_list([b[1] for b in batch])
        dset_data = Batch.from_data_list([b[2] for b in batch])
        y = torch.FloatTensor([b[3] for b in batch])
        return pipe1, pipe2, dset_data, y

    @staticmethod
    def val_collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, HeterogeneousBatch, Batch, torch.Tensor]:
        task_id = torch.LongTensor([b[0] for b in batch])
        pipe_id = torch.LongTensor([b[1] for b in batch])
        pipe = HeterogeneousBatch.from_heterogeneous_data_list([b[2] for b in batch])
        dset_data = Batch.from_data_list([b[3] for b in batch])
        y = torch.FloatTensor([b[4] for b in batch])
        return task_id, pipe_id, pipe, dset_data, y

    def __len__(self):
        return len(self.task_pipe_comb)

    def _get_train_sample(self) -> Tuple[str, str, Data, float]:
        task_id = np.random.choice(self.dataset_ids, 1).item()
        group = self.groups[task_id]
        idx1 = np.random.choice(group.index, 1).item()
        metric1 = group.loc[idx1].metric.item()
        idx2 = np.random.choice(group[group.metric != metric1].index, 1).item()
        idxes = np.asarray([idx1, idx2])
        samples = group.loc[idxes]
        metric1, metric2 = samples.metric.to_list()
        pipe1, pipe2 = samples.pipeline_id.to_list()
        with open(self.id2pipe[pipe1], "rb", os.O_NONBLOCK) as f:
            pipe1_json_string = pickle.load(f)
        with open(self.id2pipe[pipe2], "rb", os.O_NONBLOCK) as f:
            pipe2_json_string = pickle.load(f)
        ds_data = Data()
        ds_data.x = torch.tensor(self.dataset_metafeatures.loc[task_id].values, dtype=torch.float32)
        if ds_data.x.dim() < 2:
            ds_data.x = ds_data.x.view(1, -1)
        if metric1 == metric2:
            label = 0.5
        elif metric1 > metric2:
            label = 1.0
        else:
            label = 0.0
        return pipe1_json_string, pipe2_json_string, ds_data, label

    def _get_val_sample(self, idx) -> Tuple[int, int, str, Data, float]:
        sample = self.task_pipe_comb.iloc[idx]
        task_id = sample.task_id
        pipe_id = sample.pipeline_id
        with open(self.id2pipe[pipe_id], "rb", os.O_NONBLOCK) as f:
            pipe_json_string = pickle.load(f)
        metric = sample.metric
        ds_data = Data()
        ds_data.x = torch.tensor(self.dataset_metafeatures.loc[task_id].values, dtype=torch.float32)
        if ds_data.x.dim() < 2:
            ds_data.x = ds_data.x.view(1, -1)
        return task_id, pipe_id, pipe_json_string, ds_data, metric

    def __getitem__(self, idx):
        if self.is_val:
            return self._get_val_sample(idx)
        else:
            return self._get_train_sample()


class HeteroPipelineDataset(Dataset):
    # Load from no_meta_features_and_fedot_pipelines_raw
    # The folder is made with `GAMLET/scripts/generate_dataset_with_hyperparameters_raw.py`.
    def __init__(
        self,
        task_pipe_comb_file: str,
        id2pipe: Dict[int, Union[pathlib.PosixPath, str]],
        id2dataset: Dict[int, str],
        is_val: bool = False,
        encode_type: Optional[Union[str, None]] = None,
    ):
        self.task_pipe_comb = pd.read_csv(task_pipe_comb_file)
        self.groups = {k: v for k, v in self.task_pipe_comb.groupby("task_id")}
        self.id2pipe = id2pipe
        self.id2dataset = id2dataset
        self.dataset_ids = list(self.groups.keys())
        self.is_val = is_val
        if self.is_val:
            self.collate_fn = self.val_collate_fn
        else:
            self.collate_fn = self.train_collate_fn
        self.pipeline_extractor = fedot_pipeline_features_extractor.FEDOTPipelineFeaturesExtractor2(encode_type)

    @staticmethod
    def train_collate_fn(batch) -> Tuple[HeterogeneousBatch, HeterogeneousBatch, torch.Tensor]:
        pipe1 = HeterogeneousBatch.from_heterogeneous_data_list([b[0] for b in batch])
        pipe2 = HeterogeneousBatch.from_heterogeneous_data_list([b[1] for b in batch])
        y = torch.FloatTensor([b[2] for b in batch])
        return pipe1, pipe2, y

    @staticmethod
    def val_collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, HeterogeneousBatch, torch.Tensor]:
        task_id = torch.LongTensor([b[0] for b in batch])
        pipe_id = torch.LongTensor([b[1] for b in batch])
        pipe = HeterogeneousBatch.from_heterogeneous_data_list([b[2] for b in batch])
        y = torch.FloatTensor([b[3] for b in batch])
        return task_id, pipe_id, pipe, y

    def __len__(self):
        return len(self.task_pipe_comb)

    def _get_train_sample(self) -> Tuple[HeterogeneousData, HeterogeneousData, float]:
        task_id = np.random.choice(self.dataset_ids, 1).item()
        group = self.groups[task_id]
        idx1 = np.random.choice(group.index, 1).item()
        metric1 = group.loc[idx1].metric.item()
        idx2 = np.random.choice(group[group.metric != metric1].index, 1).item()
        idxes = np.asarray([idx1, idx2])
        samples = group.loc[idxes]
        metric1, metric2 = samples.metric.to_list()
        pipe1, pipe2 = samples.pipeline_id.to_list()
        with open(self.id2pipe[pipe1], "rb", os.O_NONBLOCK) as f:
            pipe1_json_string = pickle.load(f)
        with open(self.id2pipe[pipe2], "rb", os.O_NONBLOCK) as f:
            pipe2_json_string = pickle.load(f)
        if metric1 == metric2:
            label = 0.5
        elif metric1 > metric2:
            label = 1.0
        else:
            label = 0.0
        pipe1 = self.pipeline_extractor(pipe1_json_string)
        pipe2 = self.pipeline_extractor(pipe2_json_string)
        return pipe1, pipe2, label

    def _get_val_sample(self, idx) -> Tuple[int, int, HeterogeneousData, float]:
        sample = self.task_pipe_comb.iloc[idx]
        task_id = sample.task_id
        pipe_id = sample.pipeline_id
        with open(self.id2pipe[pipe_id], "rb", os.O_NONBLOCK) as f:
            pipe_json_string = pickle.load(f)
        metric = sample.metric
        pipe = self.pipeline_extractor(pipe_json_string)
        return task_id, pipe_id, pipe, metric

    def __getitem__(self, idx):
        if self.is_val:
            return self._get_val_sample(idx)
        else:
            return self._get_train_sample()
