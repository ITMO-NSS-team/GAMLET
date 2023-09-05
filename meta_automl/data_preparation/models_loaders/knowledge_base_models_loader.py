import logging
from os import PathLike
from pathlib import Path
from typing import Union, Literal, Optional, Sequence, List

import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.fitness import SingleObjFitness

from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.file_system import get_data_dir
from meta_automl.data_preparation.model import Model
from meta_automl.data_preparation.models_loaders import ModelsLoader

DEFAULT_KNOWLEDGE_BASE_PATH = get_data_dir() / 'knowledge_base_0'


class KnowledgeBaseModelsLoader(ModelsLoader):
    def __init__(self, knowledge_base_path: Union[str, PathLike] = DEFAULT_KNOWLEDGE_BASE_PATH,
                 datasets_loader: DatasetsLoader = OpenMLDatasetsLoader):
        self.knowledge_base_path: Path = Path(knowledge_base_path)
        self.df_knowledge_base: Optional[pd.DataFrame] = None
        self.df_datasets: Optional[pd.DataFrame] = None
        self.datasets_loader = datasets_loader

    def load(self, dataset_ids: Optional[Sequence[str]] = None,
             fitness_metric: str = 'f1') -> List[Model]:
        if self.df_knowledge_base is None:
            knowledge_base_split_file = self.knowledge_base_path.joinpath('knowledge_base.csv')
            self.df_knowledge_base = pd.read_csv(knowledge_base_split_file)

        if dataset_ids is None:
            dataset_ids = self.parse_datasets()['dataset_id']

        df_knowledge_base = self.df_knowledge_base
        df_knowledge_base = df_knowledge_base[df_knowledge_base['dataset_id'].isin(dataset_ids)]

        cached_datasets = {}
        for id_ in dataset_ids:
            cached_datasets[id_] = self.datasets_loader.load_single(id_)

        models = []
        for _, row in df_knowledge_base.iterrows():
            pipeline = Pipeline()
            pipeline.log.setLevel(logging.CRITICAL)
            predictor = pipeline.load(str(self.knowledge_base_path.joinpath(row['model_path'])))
            metric_value = row[fitness_metric]
            fitness = SingleObjFitness(metric_value)
            metadata = dict(row)
            dataset_cache = cached_datasets[row['dataset_id']]
            model = Model(predictor, fitness, fitness_metric, dataset_cache, metadata)
            models.append(model)
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
