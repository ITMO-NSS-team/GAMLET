import sys
sys.path.append("..")

import json
import os
import pathlib
import pickle

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from torch_geometric.data import Data

from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor
from meta_automl.data_preparation.model import Model
from meta_automl.data_preparation.models_loaders import KnowledgeBaseModelsLoader
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor



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
        knowledge_base_directory: str,
        dataset_directory: str,
        meta_features_extractor: MetaFeaturesExtractor,
        split: Optional[str] = "all",  # Can be train, test, all
        train_test_split_name: Optional[str] = "train_test_datasets_classification.csv",
        task_type: Optional[str] = "classification",
        fitness_metric: Optional[str] = "f1",
        exclude_datasets: Optional[List[str]] = [],
        meta_features_preprocessors: Dict[str, Any] = None,
        use_hyperpar: bool = False,
        models_loader_kwargs: Dict[str, Any] = {},
    ) -> None:
        if task_type != "classification":
            raise NotImplementedError(f"Current version if for `'classification'` `task_type`")

        self.knowledge_base_directory = knowledge_base_directory
        self.dataset_directory = dataset_directory
        self.train_test_split_name = train_test_split_name
        self.task_type = task_type
        self.split = split
        self.fitness_metric = fitness_metric
        self.exclude_datasets = exclude_datasets
        self.meta_features_preprocessors = meta_features_preprocessors

        self._maybe_create_dataset_directory(os.path.join(self.dataset_directory, self.split))

        self.pipeline_extractor = FEDOTPipelineFeaturesExtractor()
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

    def _maybe_create_dataset_directory(self, directory: str) -> None:
        if not os.path.exists(directory):
            pathlib.Path(directory).mkdir(parents=True)

    def _get_best_pipelines_unique_indexes(self, dataset_models: List[Model]) -> List[int]:
        raise NotImplementedError("Broken code in the function.")
        # best_pipelines_unique_indexes = temp_df.groupby('pipeline_id')['fitness'].max().reset_index()
        # return best_pipelines_unique_indexes

    def _process(self) -> Tuple[List[Dict[str, Union[float, int]]], List[Dict[str, float]], List[int]]:
        df_dataset_models = pd.DataFrame(self.models_loader.load(fitness_metric=self.fitness_metric))
        df_dataset_models['task_id'] = df_dataset_models.metadata.apply(lambda x: x['dataset_id'])
        df_dataset_models['y'] = df_dataset_models['fitness'].astype(float)

        if self.use_hyperpar:
            df_dataset_models['pipeline_id'] = df_dataset_models.index  # or create another hash?
            task_pipe_comb = df_dataset_models
        else:
            df_dataset_models['pipeline_hash'] = df_dataset_models.predictor.apply(calc_pipeline_hash).astype(str)
            task_pipe_comb = df_dataset_models.groupby(['task_id', 'pipeline_hash'])['y'].max().reset_index()
            idxes = df_dataset_models.reset_index().groupby(['task_id', 'pipeline_hash'])['y'].idxmax()
            task_pipe_comb = df_dataset_models.loc[idxes]
            codes, _ = pd.factorize(task_pipe_comb['pipeline_hash'])
            task_pipe_comb['pipeline_id'] = codes
        pipelines_fedot = task_pipe_comb.drop_duplicates(subset=['pipeline_id']) \
                                        .sort_values(by=['pipeline_id'])['predictor'] \
                                        .values.tolist()
        pipeline_extractor = FEDOTPipelineFeaturesExtractor(include_operations_hyperparameters=False,
                                                            operation_encoding="ordinal")
        pipelines_data = [get_pipeline_features(pipeline_extractor, pl) for pl in pipelines_fedot]
        return (
            task_pipe_comb[["task_id", "pipeline_id", "y"]],
            pipelines_fedot,
            pipelines_data,
            self.df_datasets[['dataset_id', 'is_train']].set_index('dataset_id')['is_train'].to_dict(),
        )

    def _save_task_pipe_comb(self, task_pipe_comb: List[Dict[str, Union[float, int]]]):
        # task_pipe_comb_df = pd.DataFrame.from_records(task_pipe_comb)
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

    def _save_split(self, is_train_flags: List[int]):
        split = {
            "train": [],
            "test": [],
        }
        for key, flag in is_train_flags.items():
            if flag == 1:
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
