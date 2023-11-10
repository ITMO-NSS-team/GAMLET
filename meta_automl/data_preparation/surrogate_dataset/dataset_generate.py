import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from torch_geometric.data import Data

from meta_automl.data_preparation.dataset import (CustomDataset,
                                                  DataNotFoundError,
                                                  DatasetData, DatasetIDType)
from meta_automl.data_preparation.file_system.file_system import ensure_dir_exists
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor
from meta_automl.data_preparation.models_loaders import KnowledgeBaseModelsLoader
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor


def dataset_from_id_without_data_loading(dataset_id: DatasetIDType) -> CustomDataset:
    """ Creates the CustomDataset object without loading the data. Use if your don't need the models
    to load the datasets data into memory, or if you have loaded the cache manually. """
    return CustomDataset(dataset_id)


def dataset_from_id_with_data_loading(dataset_id: DatasetIDType) -> CustomDataset:
    """ Load dataset from '//10.9.14.114/calc/Nikitin/datasets/' into the project cache directory.
    As a result, every model of the knowledge base will have its data available by
    `model.dataset.get_data()`.
    """
    dataset = CustomDataset(dataset_id)
    try:
        dataset.get_data()
    except DataNotFoundError:
        data_root = '//10.9.14.114/calc/Nikitin/datasets/'
        dataset_name, fold_num = dataset_id[:-2], dataset_id[-1]
        data_path = f'{dataset_name}_fold{fold_num}.npy'
        data_x = []
        for path_prefix in ('train_', 'test_'):
            data_x.append(np.load(data_root + path_prefix + data_path))
        data_y = []
        for path_prefix in ('trainy_', 'testy_'):
            data_y.append(np.load(data_root + path_prefix + data_path))
        data_x = np.concatenate(data_x)
        data_y = np.concatenate(data_y)
        data_x = pd.DataFrame(data_x)
        data_y = pd.DataFrame(data_y)
        dataset_data = DatasetData(data_x, data_y)
        dataset.dump_data(dataset_data)
    return dataset


def calc_pipeline_hash(pl: Pipeline) -> str:
    ''' not real hash'''
    edges_str = " ".join([",".join([str(item) for item in sublist]) for sublist in pl.get_edges()])
    nodes_str = ' '.join([str(item) for item in pl.nodes])
    return edges_str + ' ' + nodes_str


def get_pipeline_features(pipeline_extractor: FEDOTPipelineFeaturesExtractor,
                          pipeline: Pipeline) -> Data:
    pipeline_json_string = pipeline.save()[0].encode()
    return pipeline_extractor(pipeline_json_string)


class KnowledgeBaseToDataset:
    def __init__(
            self,
            knowledge_base_directory: os.PathLike,
            dataset_directory: os.PathLike,
            meta_features_extractor: MetaFeaturesExtractor,
            split: Literal['train', 'test', 'all'] = 'all',
            train_test_split_name: Optional[str] = "train_test_datasets_classification.csv",
            task_type: Optional[str] = "classification",
            fitness_metric: Optional[str] = "f1",
            exclude_datasets=None,
            meta_features_preprocessors: Dict[str, Any] = None,
            use_hyperpar: bool = False,
            models_loader_kwargs=None,
    ) -> None:
        if exclude_datasets is None:
            exclude_datasets = []
        if models_loader_kwargs is None:
            models_loader_kwargs = {}
        if task_type != "classification":
            raise NotImplementedError("Current version is for `task_type='classification'`")

        self.knowledge_base_directory = knowledge_base_directory
        self.dataset_directory = dataset_directory
        self.train_test_split_name = train_test_split_name
        self.task_type = task_type
        self.split = split
        self.fitness_metric = fitness_metric
        self.exclude_datasets = exclude_datasets
        self.meta_features_preprocessors = meta_features_preprocessors

        ensure_dir_exists(Path(self.dataset_directory, self.split))

        self.pipeline_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False,
                                                                 operation_encoding="ordinal")
        self.meta_features_extractor = meta_features_extractor

        self.models_loader = KnowledgeBaseModelsLoader(self.knowledge_base_directory, **models_loader_kwargs)
        df_datasets = self.models_loader.parse_datasets(self.split, self.task_type)
        self.df_datasets = df_datasets[df_datasets["dataset_name"].apply(lambda x: x not in self.exclude_datasets)]

        self.use_hyperpar = use_hyperpar

    def _check_for_duplicated_datasets(self):
        occurences = self.df_datasets.dataset_id.value_counts()
        unique_number_of_occurences = set(occurences.to_list())
        assert len(unique_number_of_occurences) == 1, f"Duplicated datasets detected. Check datasets: \n{occurences}"
        assert unique_number_of_occurences.pop() == 1, f"Duplicated datasets detected. Check datasets: \n{occurences}"

    def _process(self) -> Tuple[pd.DataFrame, List[Dict[str, float]], List[Data], Dict[DatasetIDType, bool]]:
        df_dataset_models = pd.DataFrame(self.models_loader.load(fitness_metric=self.fitness_metric))
        df_dataset_models['task_id'] = df_dataset_models.metadata.apply(lambda x: x['dataset_id'])
        df_dataset_models['y'] = df_dataset_models['fitness'].astype(float)

        if self.use_hyperpar:
            df_dataset_models['pipeline_id'] = df_dataset_models.index  # or create another hash?
            task_pipe_comb = df_dataset_models
        else:
            df_dataset_models['pipeline_hash'] = df_dataset_models.predictor.apply(calc_pipeline_hash).astype(str)
            idxes = df_dataset_models.reset_index().groupby(['task_id', 'pipeline_hash'])['y'].idxmax()
            task_pipe_comb = df_dataset_models.loc[idxes]
            codes, _ = pd.factorize(task_pipe_comb['pipeline_hash'])
            task_pipe_comb['pipeline_id'] = codes
        pipelines_fedot = task_pipe_comb.drop_duplicates(subset=['pipeline_id']) \
            .sort_values(by=['pipeline_id'])['predictor'] \
            .values.tolist()

        pipelines_data = [get_pipeline_features(self.pipeline_extractor, pl) for pl in pipelines_fedot]
        return (
            task_pipe_comb[["task_id", "pipeline_id", "y"]],
            pipelines_fedot,
            pipelines_data,
            self.df_datasets[['dataset_id', 'is_train']].set_index('dataset_id')['is_train'].to_dict(),
        )

    def _save_task_pipe_comb(self, task_pipe_comb: pd.DataFrame):
        task_pipe_comb.to_csv(
            os.path.join(self.dataset_directory, self.split, "task_pipe_comb.csv"),
            index=False,
        )

    def _save_datasets_meta_features(self, datasets_meta_features):
        if self.meta_features_preprocessors is not None:
            # df_as_dict = {k: list(v.values()) for k, v in df.to_dict().items()}
            self.meta_features_preprocessors.fit(
                datasets_meta_features,
                os.path.join(self.dataset_directory, self.split, "meta_features_preprocessors.pickle"),
            )
            transformed = self.meta_features_preprocessors.transform(datasets_meta_features, single=False)
            # df = pd.DataFrame.from_dict({k: v.reshape(-1) for k,v in transformed.items()})
        transformed = transformed.groupby(by=['dataset', 'variable'])['value'].apply(list).apply(lambda x: pd.Series(x))
        transformed.to_csv(
            os.path.join(self.dataset_directory, self.split, "datasets.csv"),
            header=True,
            index=False,
        )

    def _save_pipelines_objects(self, pipelines: List[Any]):
        with open(os.path.join(self.dataset_directory, self.split, "pipelines_fedot.pickle"), "wb") as f:
            pickle.dump(pipelines, f)

    def _save_pipelines_data(self, pipelines: List[Any]):
        with open(os.path.join(self.dataset_directory, self.split, "pipelines.pickle"), "wb") as f:
            pickle.dump(pipelines, f)

    def _save_split(self, is_train_flags: Dict[DatasetIDType, bool]):
        split = {
            "train": [],
            "test": [],
        }
        for key, flag in is_train_flags.items():
            if flag:
                split["train"].append(key)
            else:
                split["test"].append(key)
        with open(os.path.join(self.dataset_directory, self.split, "split.json"), "w") as f:
            json.dump(split, f)

    def convert_pipelines(self):
        task_pipe_comb, pipelines_fedot, pipelines_data, is_train_flags = self._process()

        self._save_split(is_train_flags)
        self._save_pipelines_objects(pipelines_fedot)
        self._save_pipelines_data(pipelines_data)
        self._save_task_pipe_comb(task_pipe_comb)

    def convert_datasets(self):
        datasets_meta_features = self.meta_features_extractor.extract(
            self.df_datasets['dataset_id'].values.tolist(),
            fill_input_nans=True,
        )
        # For PyMFE. OpenML provides a dictionary of floats.
        # if isinstance(datasets_meta_features[0], pd.DataFrame):
        #     datasets_meta_features = [df.iloc[0].to_dict() for df in datasets_meta_features]
        self._save_datasets_meta_features(datasets_meta_features)
