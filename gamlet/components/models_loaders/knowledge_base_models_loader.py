import logging
import os
from functools import partial
from itertools import chain
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from typing_extensions import Literal

from gamlet.components.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from gamlet.components.models_loaders import ModelsLoader
from gamlet.data_preparation.dataset import TimeSeriesDataset
from gamlet.data_preparation.evaluated_model import EvaluatedModel
from gamlet.data_preparation.file_system import get_data_dir, get_project_root

DEFAULT_KNOWLEDGE_BASE_PATH = get_data_dir() / 'knowledge_base_0'


def parallelize(data, func, num_workers):
    data_split = np.array_split(data, num_workers)
    pool = Pool(num_workers)
    data = list(chain(*pool.map(func, data_split)))
    pool.close()
    pool.join()
    return data


def process_record(df, knowledge_base_path, fitness_metric: str):
    models = []
    for _, row in df.iterrows():
        json_path = knowledge_base_path.joinpath(PureWindowsPath(row['model_path']))
        pipeline = Pipeline()
        pipeline.log.setLevel(logging.CRITICAL)
        predictor = pipeline.load(json_path)

        metric_value = row[fitness_metric]
        # fitness = SingleObjFitness(metric_value)
        metadata = dict(row)
        models.append(EvaluatedModel(predictor, metric_value, fitness_metric, None, metadata))  # row['dataset_cache']
    return models


def read_history(d_ids, knowledge_base_path):
    models = []
    adapter = PipelineAdapter()
    for d_id in d_ids:
        try:
            history = OptHistory().load(knowledge_base_path / d_id / 'opt_history.json')
        except Exception:
            continue

        for gen in history.generations:
            for ind in gen:
                pipeline = adapter.restore(ind.graph)
                metadata = {'dataset_id': d_id}
                models.append(EvaluatedModel(pipeline, -1 * ind.fitness.value, history.objective.metric_names[0],
                                             TimeSeriesDataset(d_id), metadata))
    return models


class KnowledgeBaseModelsLoader(ModelsLoader):
    def __init__(
            self,
            knowledge_base_path: Union[str, PathLike] = DEFAULT_KNOWLEDGE_BASE_PATH,
            datasets_loader: DatasetsLoader = OpenMLDatasetsLoader(),
    ):
        self.knowledge_base_path: Path = Path(knowledge_base_path)
        self.df_knowledge_base: Optional[pd.DataFrame] = None
        self.df_datasets: Optional[pd.DataFrame] = None
        self.datasets_loader = datasets_loader

    def load(
            self,
            dataset_ids: Optional[Sequence[str]] = None, fitness_metric: str = 'fitness',
    ) -> List[EvaluatedModel]:
        if self.df_knowledge_base is None:
            knowledge_base_split_file = self.knowledge_base_path.joinpath('knowledge_base.csv')
            self.df_knowledge_base = pd.read_csv(knowledge_base_split_file)

        if dataset_ids is None:
            dataset_ids = self.load_dataset_split()['dataset_id']

        self.df_knowledge_base['fitness_coef'] = -1
        self.df_knowledge_base[fitness_metric] *= self.df_knowledge_base['fitness_coef']

        df_knowledge_base = self.df_knowledge_base
        df_knowledge_base = df_knowledge_base[df_knowledge_base['dataset_id'].isin(dataset_ids)]

        partitions = max(cpu_count() - 2, 1)
        models = parallelize(df_knowledge_base,
                             partial(process_record, knowledge_base_path=self.knowledge_base_path,
                                     fitness_metric=fitness_metric),
                             num_workers=partitions)
        return models

    def load_dataset_split(
            self, train_test: Literal['train', 'test', 'all'] = 'all',
            task_type: Literal['classification', 'regression', 'ts_forecasting'] = 'classification') -> pd.DataFrame:
        if self.df_datasets is None:
            train_test_split_file = f'train_test_datasets_{task_type}.csv'
            train_test_split_file = self.knowledge_base_path.joinpath(train_test_split_file)
            self.df_datasets = pd.read_csv(train_test_split_file)

        df_datasets = self.df_datasets
        if train_test in ('train', 'test'):
            is_train = (train_test == 'train')
            df_datasets = df_datasets[df_datasets['is_train'] == is_train]
            df_datasets = df_datasets.drop(columns='is_train')

        return df_datasets


class KBTSModelsLoader(ModelsLoader):
    def __init__(self,
                 kb_path,
                 split_f_name='train_test_datasets.csv'):
        self.knowledge_base_path = get_project_root() / kb_path / 'datasets'
        self.split_file = get_project_root() / kb_path / split_f_name
        self.df_datasets = None

    def load(
            self,
            dataset_ids: Optional[Sequence[str]] = None,
    ) -> List[EvaluatedModel]:
        if dataset_ids is None:
            dataset_ids = os.listdir(self.knowledge_base_path)
            assert len(dataset_ids) != 0, "Datasets not found"
        partitions = max(cpu_count() - 2, 1)
        get_models = partial(read_history, knowledge_base_path=self.knowledge_base_path)
        models = parallelize(dataset_ids, get_models,
                             num_workers=partitions)
        return models

    def _create_split(self,
                      test_ratio=0.3,
                      seed=0):
        rng = np.random.default_rng(seed=seed)
        keys = os.listdir(self.knowledge_base_path)
        n = len(keys)
        label = np.ones(n)
        label[:int(n * test_ratio)] = 0
        rng.shuffle(label)
        df_split = pd.DataFrame({'dataset_id': keys, 'dataset_name': keys, 'is_train': label.astype(int)})
        df_split.to_csv(self.split_file, index=False)

    def load_dataset_split(
            self, train_test: Literal['train', 'test', 'all'] = 'all',
            task_type: Literal['classification', 'regression', 'ts_forecasting'] = 'classification') -> pd.DataFrame:

        if not self.split_file.is_file():
            self._create_split()

        self.df_datasets = pd.read_csv(self.split_file)
        df_datasets = self.df_datasets
        if train_test in ('train', 'test'):
            is_train = (train_test == 'train')
            df_datasets = df_datasets[df_datasets['is_train'] == is_train]
            df_datasets = df_datasets.drop(columns='is_train')
        return df_datasets
