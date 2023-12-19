import logging
from functools import partial
from itertools import chain
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from typing_extensions import Literal

from gamlet.components.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from gamlet.components.models_loaders import ModelsLoader
from gamlet.data_preparation.evaluated_model import EvaluatedModel
from gamlet.data_preparation.file_system import get_data_dir

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
        models.append(EvaluatedModel(predictor, metric_value, 'fitness', row['dataset_cache'], metadata))
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
            dataset_ids: Optional[Sequence[str]] = None,
            fitness_metric: str = 'f1',
    ) -> List[EvaluatedModel]:
        if self.df_knowledge_base is None:
            knowledge_base_split_file = self.knowledge_base_path.joinpath('knowledge_base.csv')
            self.df_knowledge_base = pd.read_csv(knowledge_base_split_file)

        if dataset_ids is None:
            dataset_ids = self.parse_datasets()['dataset_id']

        self.df_knowledge_base['fitness_coef'] = -1
        self.df_knowledge_base[fitness_metric] *= self.df_knowledge_base['fitness_coef']

        df_knowledge_base = self.df_knowledge_base
        df_knowledge_base = df_knowledge_base[df_knowledge_base['dataset_id'].isin(dataset_ids)]

        cached_datasets = {}
        for id_ in dataset_ids:
            cached_datasets[id_] = self.datasets_loader.load_single(id_)
        df_knowledge_base['dataset_cache'] = df_knowledge_base['dataset_id'].map(cached_datasets)

        partitions = max(cpu_count() - 2, 1)
        models = parallelize(df_knowledge_base,
                             partial(process_record, knowledge_base_path=self.knowledge_base_path,
                                     fitness_metric=fitness_metric),
                             num_workers=partitions)
        return models

    def parse_datasets(
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
