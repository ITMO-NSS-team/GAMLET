import sys
sys.path.append("..")

import pickle
import shutil
from tqdm import tqdm
from typing import List, Union, Tuple, Dict, Optional, Any
import json
import pickle
import os
import openml
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pathlib
from fedot.core.pipelines.pipeline import Pipeline

from meta_automl.data_preparation.models_loaders import KnowledgeBaseModelsLoader
from meta_automl.data_preparation.pipeline_features_extractors import FEDOTPipelineFeaturesExtractor
from meta_automl.data_preparation.feature_preprocessors import FeaturesPreprocessor
from meta_automl.data_preparation.model import Model
from meta_automl.data_preparation.meta_features_extractors import MetaFeaturesExtractor

def calc_pipeline_hash(pl: Pipeline)-> Tuple[str]:
    return tuple(str(pl.get_edges())+str(pl.nodes))


class KnowledgeBaseToDataset:
    def __init__(
        self,
        knowledge_base_directory: str,
        dataset_directory: str,
        meta_features_extractor: MetaFeaturesExtractor,
        split: Optional[str] = "all", # Can be train, test, all
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
        if fitness_metric == "log_loss":
            self.fitness_coef = 1
        else:
            self.fitness_coef = -1

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

        self._check_for_duplicated_datasets()

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
        temp_df = pd.DataFrame(columns=["predictor", "fitness"])
        temp_df["predictor"] = [calc_pipeline_hash(x.predictor) for x in dataset_models]
        temp_df["fitness"] = [self.fitness_coef * x.fitness.value for x in dataset_models]
        # Select top-1 pipeline
        best_pipelines_unique_indexes = temp_df.groupby('predictor')['fitness'].idxmax().to_list()
        return best_pipelines_unique_indexes

    def _process(self) -> Tuple[List[Dict[str, Union[float, int]]], List[Dict[str, float]], List[int]]:
        pipeline_id = 0
        task_pipe_comb = []
        datasets_meta_features = []
        pipelines = []
        dict_pipelines = dict()
        is_train_flags = []

        for task_id in tqdm(self.df_datasets.index):
            dataset = self.df_datasets.loc[task_id]
            # datasets_meta_features.append(self.meta_features_extractor(dataset.dataset_id))
            try:
                dataset_id = int(dataset.dataset_id)
            except ValueError:
                dataset_id = dataset.dataset_id
            datasets_meta_features.append(self.meta_features_extractor.extract([dataset_id], fill_input_nans=True))
            is_train_flags.append(dataset.is_train)

            dataset_models = self.models_loader.load(
                dataset_ids=[dataset_id],
                fitness_metric=self.fitness_metric,
            )

            if self.use_hyperpar:
                best_pipelines_unique_indexes = list(range(len(dataset_models)))
            else:
                best_pipelines_unique_indexes = self._get_best_pipelines_unique_indexes(dataset_models)

            for index in best_pipelines_unique_indexes:
                model = dataset_models[index]
                pipeline_hash = calc_pipeline_hash(model.predictor)
                if pipeline_hash not in dict_pipelines:
                    dict_pipelines[pipeline_hash] = pipeline_id
                    pipelines.append(model.predictor)
                    pipeline_id += 1

                y = self.fitness_coef * model.fitness.value
                task_pipe_comb.append({"task_id": task_id, "pipeline_id": dict_pipelines[pipeline_hash], "y": y})

        # For PyMFE. OpenML provides a dictionary of floats.
        if isinstance(datasets_meta_features[0], pd.DataFrame):
            datasets_meta_features = [df.iloc[0].to_dict() for df in datasets_meta_features]

        return task_pipe_comb, datasets_meta_features, pipelines, is_train_flags

    def _save_task_pipe_comb(self, task_pipe_comb: List[Dict[str, Union[float, int]]]):
        task_pipe_comb_df = pd.DataFrame.from_records(task_pipe_comb)
        task_pipe_comb_df.to_csv(
            os.path.join(self.dataset_directory, self.split, "task_pipe_comb.csv"),
            header=True,
            index=True,
        )

    def _save_datasets_meta_features(self, datasets_meta_features: List[Dict[str, float]]):
        df = pd.DataFrame.from_records(datasets_meta_features)
        if self.meta_features_preprocessors is not None:
            df_as_dict = {k: list(v.values()) for k, v in df.to_dict().items()}
            self.meta_features_preprocessors.fit(
                df_as_dict,
                os.path.join(self.dataset_directory, self.split, "meta_features_preprocessors.pickle"),
            )
            transformed = self.meta_features_preprocessors.transform(df_as_dict, single=False)
            df = pd.DataFrame.from_dict({k: v.reshape(-1) for k,v in transformed.items()})

        df.to_csv(
            os.path.join(self.dataset_directory, self.split, "datasets.csv"),
            header=True,
            index=False,
        )

    def _save_pipelines_objects(self, pipelines: List[Any]):
        with open(os.path.join(self.dataset_directory, self.split, "pipelines_fedot.pickle"), "wb") as f:
            pickle.dump(pipelines, f)

    def _save_split(self, is_train_flags: List[int]):
        split = {
            "train": [],
            "test": [],
        }
        for i, flag in enumerate(is_train_flags):
            if flag == 1:
                split["train"].append(i)
            else:
                split["test"].append(i)
        with open(os.path.join(self.dataset_directory, self.split, "split.json"), "w") as f:
            json.dump(split, f)

    def convert(self):
        task_pipe_comb, datasets_meta_features, pipelines, is_train_flags = self._process()

        self._save_split(is_train_flags)
        self._save_pipelines_objects(pipelines)
        self._save_datasets_meta_features(datasets_meta_features)
        self._save_task_pipe_comb(task_pipe_comb)