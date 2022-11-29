from abc import abstractmethod
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence
from fedot.core.validation.split import tabular_cv_generator
from sklearn.metrics import roc_auc_score as roc_auc

from components.data_preparation.dataset import DatasetCache
from support.data_utils import PathType


class ModelSelector:

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def select(self, *args, **kwargs):
        raise NotImplementedError()


def get_best_fedot_performers(dataset: DatasetCache, pipelines: List[Pipeline], fit_from_scratch: bool = False,
                              n_best: int = 1) -> Union[Pipeline, List[Pipeline]]:
    loaded_dataset = dataset.load()
    X, y_test = loaded_dataset.X, loaded_dataset.y
    X = InputData(idx=np.arange(0, len(X)), features=X, target=y_test, data_type=DataTypesEnum.table,
                  task=Task(TaskTypesEnum.classification))
    metrics_values = []
    for pipeline in pipelines:
        if not pipeline.is_fitted or fit_from_scratch:
            pipeline.unfit()
            cv_folds = partial(tabular_cv_generator, X, folds=5)
            objective_eval = PipelineObjectiveEvaluate(MetricsObjective(ClassificationMetricsEnum.ROCAUC), cv_folds)
            metric_value = objective_eval(pipeline).value
        else:
            predict_labels = pipeline.predict(X)
            y_pred = predict_labels.predict
            metric_value = roc_auc(y_test, y_pred)
        metrics_values.append(metric_value)
    if n_best == 1:
        best_pipeline = pipelines[np.argmax(metrics_values)]
    else:
        best_pipeline = [pipelines.pop(np.argmax(metrics_values)) for _ in range(min(n_best, len(pipelines)))]
    return best_pipeline


class FedotResultsBestPipelineSelector(ModelSelector):
    def __init__(self):
        self.datasets: Optional[List[DatasetCache]] = None
        self.selected_models: Optional[List[Union[Pipeline, List[Pipeline]]]] = None
        self.pipeline_paths: Optional[List[Union[PathType, List[PathType]]]] = None
        self.launch_dir: Optional[PathType] = None

    def fit(self, datasets: List[DatasetCache], pipeline_paths: Optional[List[Union[PathType, List[PathType]]]] = None,
            launch_dir: Optional[PathType] = None):
        self.datasets = datasets
        self.pipeline_paths = pipeline_paths
        self.launch_dir = launch_dir
        if not self.pipeline_paths:
            self.define_model_paths()
        return self

    def select(self, n_best: int = 1, fit_from_scratch: bool = False):
        pipelines = []
        for dataset, path in zip(self.datasets, self.pipeline_paths):
            if isinstance(path, list):
                candidate_pipelines = [Pipeline.from_serialized(str(p)) for p in path]
                pipeline = get_best_fedot_performers(dataset, candidate_pipelines, fit_from_scratch, n_best)
            else:
                pipeline = Pipeline.from_serialized(str(path))
            if n_best > 1:
                ensure_wrapped_in_sequence(pipeline)
            pipelines.append(pipeline)
        self.selected_models = pipelines
        return pipelines

    def define_model_paths(self):
        if not self.launch_dir:
            raise ValueError('Launch dir or pipeline paths must be provided!')

        launch_dir = self.launch_dir
        if isinstance(launch_dir, str):
            launch_dir = Path(launch_dir)
        datasets_launch_dates = {}
        datasets_launch_dates_str = {}

        for dataset in self.datasets:
            dataset_name = dataset.name
            for launch in launch_dir.glob(f'{dataset_name}\\FEDOT\\*\\launch_0'):
                launch_date_dir = launch.parent
                # launch_date_dir: airlines\FEDOT\07-07-2022-08-25-07
                dataset_name = launch_date_dir.parents[1].name
                launch_date_str = launch_date_dir.name
                launch_date = datetime.strptime(launch_date_str, '%d-%m-%Y-%H-%M-%S')

                if launch_date >= datasets_launch_dates.get(dataset_name, launch_date):
                    datasets_launch_dates_str[dataset_name] = launch_date_str
                    datasets_launch_dates[dataset_name] = launch_date

        if len(datasets_launch_dates_str) != len(self.datasets):
            raise ValueError('FEDOT launches not found!')

        datasets_models_paths = dict(zip(datasets_launch_dates_str.keys(), [[]] * len(datasets_launch_dates_str)))

        for dataset_name, launch_date_str in datasets_launch_dates_str.items():
            launches_path = Path(launch_dir, dataset_name, 'FEDOT', launch_date_str)
            for model_path in launches_path.glob(r'*\launch_*.json'):
                datasets_models_paths[dataset_name].append(model_path)

        self.pipeline_paths = [datasets_models_paths[dataset.name] for dataset in self.datasets]
        return self


# class OpenMLSelector(ModelSelector):
#     def __init__(self):
#         self.dataset_ids: Optional[List[DatasetCache]] = None
#         self.selected_models: Optional[List[Union[Pipeline, List[Pipeline]]]] = None
#
#     def fit(self, dataset_ids: List[OpenMLDatasetID], select_best: bool = True, n_models: int = 1,
#             suitability_filter: Callable[[OpenMLEvaluation], bool] = lambda e: True):
#         self.dataset_ids = dataset_ids
#         # self.pipeline_paths = pipeline_paths
#         # self.launch_dir = launch_dir
#         # if not self.pipeline_paths:
#         #     self.define_model_paths()
#         return self
#
#     def select(self, n_best: int = 1, fit_from_scratch: bool = False):
#         pipelines = []
#         for ... in ...:
#             pipeline = ...
#             if n_best > 1:
#                 ensure_wrapped_in_sequence(pipeline)
#             pipelines.append(pipeline)
#         self.selected_models = pipelines
#         return pipelines
