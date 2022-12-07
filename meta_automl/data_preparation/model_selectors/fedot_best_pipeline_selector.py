from functools import partial
from pathlib import Path
from typing import List, Union, Optional

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
from tqdm import tqdm

from meta_automl.data_preparation.data_directory_manager import PathType
from meta_automl.data_preparation.dataset import DatasetCache, NoCacheError
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.model_selectors import ModelSelector


def evaluate_classification_fedot_pipeline(pipeline, input_data):
    cv_folds = partial(tabular_cv_generator, input_data, folds=5)
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(ClassificationMetricsEnum.ROCAUC), cv_folds)
    metric_value = objective_eval(pipeline).value
    return metric_value


def get_best_fedot_performers(dataset: DatasetCache, pipelines: List[Pipeline], datasets_loader: DatasetsLoader,
                              n_best: int = 1) -> Union[Pipeline, List[Pipeline]]:
    try:
        loaded_dataset = dataset.load_into_memory()
    except NoCacheError:
        loaded_dataset = datasets_loader.load_single(dataset.name).load_into_memory()
    X, y_test = loaded_dataset.X, loaded_dataset.y
    input_data = InputData(idx=np.arange(0, len(X)), features=X, target=y_test, data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    metric_values = []
    for pipeline in pipelines:
        metric_value = evaluate_classification_fedot_pipeline(pipeline, input_data)
        metric_values.append(metric_value)
    if n_best == 1:
        best_pipeline = pipelines[np.argmax(metric_values)]
    else:
        best_pipeline = [pipelines.pop(np.argmax(metric_values)) for _ in range(min(n_best, len(pipelines)))]
    return best_pipeline


class FEDOTResultsBestPipelineSelector(ModelSelector):
    def __init__(self):
        self.datasets: Optional[List[DatasetCache]] = None
        self.selected_models: Optional[List[Union[Pipeline, List[Pipeline]]]] = None
        self.pipeline_paths: Optional[List[Union[PathType, List[PathType]]]] = None
        self.launch_dir: Optional[PathType] = None
        self.datasets_loader: Optional[DatasetsLoader] = None

    def fit(self, datasets: Union[List[Union[DatasetCache, str]], str] = 'all',
            pipeline_paths: Optional[List[Union[PathType, List[PathType]]]] = None,
            launch_dir: Optional[PathType] = None, datasets_loader: Optional[DatasetsLoader] = None):

        self.launch_dir: Path = Path(launch_dir) if isinstance(launch_dir, str) else launch_dir
        self.datasets_loader = datasets_loader or OpenMLDatasetsLoader()

        self.datasets: List[DatasetCache] = (self._define_datasets() if datasets == 'all'
                                             else self._dataset_names_to_cache(datasets))

        self.pipeline_paths = pipeline_paths or self._define_model_paths()
        return self

    def select(self, n_best: int = 1):
        pipelines = []
        for dataset, path in tqdm(list(zip(self.datasets, self.pipeline_paths)), desc='Selecting best models',
                                  leave=False):
            if isinstance(path, list):
                candidate_pipelines = [Pipeline.from_serialized(str(p)) for p in tqdm(path, desc='Importing pipelines',
                                                                                      leave=False)]
                pipeline = get_best_fedot_performers(dataset, candidate_pipelines, self.datasets_loader, n_best)
            else:
                pipeline = Pipeline.from_serialized(str(path))
            if n_best > 1:
                ensure_wrapped_in_sequence(pipeline)
            pipelines.append(pipeline)
        self.selected_models = pipelines
        return pipelines

    def _define_datasets(self) -> List[DatasetCache]:
        if not self.launch_dir:
            raise ValueError('Launch dir or datasets must be provided!')

        datasets = list({p.parents[2].name for p in self.launch_dir.glob(r'*\FEDOT*\*\launch_0')})
        datasets.sort()
        datasets = self._dataset_names_to_cache(datasets)
        return datasets

    def _define_model_paths(self) -> List[List[Path]]:
        if not self.launch_dir:
            raise ValueError('Launch dir or model paths must be provided!')

        dataset_names = self.dataset_names
        datasets_models_paths = dict(zip(dataset_names, [[]] * len(dataset_names)))

        for dataset_name in tqdm(dataset_names, desc='Defining model paths'):
            for model_path in self.launch_dir.glob(f'{dataset_name}\\FEDOT*\\*\\*\\launch_*.json'):
                datasets_models_paths[dataset_name].append(model_path)

        return [datasets_models_paths[dataset.name] for dataset in self.datasets]

    @property
    def dataset_names(self):
        return [d.name if isinstance(d, DatasetCache) else d for d in self.datasets]

    @staticmethod
    def _dataset_names_to_cache(datasets: List[Union[str, DatasetCache]]) -> List[DatasetCache]:
        new_list = []
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset = DatasetCache(dataset)
            new_list.append(dataset)
        return new_list
