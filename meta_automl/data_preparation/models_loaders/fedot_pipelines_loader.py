from functools import partial
from pathlib import Path
from typing import List, Union, Optional, Literal

import numpy as np
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator
from golem.core.log import default_log
from tqdm import tqdm

from meta_automl.data_preparation.data_manager import PathType
from meta_automl.data_preparation.dataset import DatasetCache
from meta_automl.data_preparation.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from meta_automl.data_preparation.model import Model
from meta_automl.data_preparation.models_loaders import ModelsLoader


def evaluate_classification_fedot_pipeline(pipeline, input_data):
    cv_folds = partial(tabular_cv_generator, input_data, folds=5)
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(ClassificationMetricsEnum.ROCAUC), cv_folds,
                                               eval_n_jobs=-1)
    fitness = objective_eval(pipeline)
    return fitness


def get_n_best_fedot_performers(dataset_cache: DatasetCache, pipelines: List[Pipeline], datasets_loader: DatasetsLoader,
                                n_best: int = 1) -> List[Model]:
    loaded_dataset = datasets_loader.cache_to_memory(dataset_cache)
    X, y_test = loaded_dataset.x, loaded_dataset.y
    input_data = InputData(idx=np.arange(0, len(X)), features=X, target=y_test, data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    fitnesses = []
    models = []
    metric_name = 'roc_auc'
    for pipeline in tqdm(pipelines, desc='Evaluating pipelines'):
        fitness = evaluate_classification_fedot_pipeline(pipeline, input_data)
        fitnesses.append(fitness)
        models.append(Model(pipeline, fitness, metric_name, dataset_cache))

    best_models = [models.pop(np.argmax(fitnesses)) for _ in range(min(n_best, len(pipelines)))]
    return best_models


class FEDOTPipelinesLoader(ModelsLoader):
    def __init__(self, datasets_to_load: Union[List[Union[DatasetCache, str]], Literal['auto']] = 'auto',
                 candidate_pipelines: Optional[List[List[Pipeline]]] = None,
                 candidate_pipeline_paths: Optional[List[List[PathType]]] = None,
                 launch_dir: Optional[PathType] = None,
                 datasets_loader: Optional[DatasetsLoader] = None):

        self.log = default_log(self)

        self.datasets_loader = datasets_loader or OpenMLDatasetsLoader()

        self.launch_dir: Path = Path(launch_dir) if isinstance(launch_dir, str) else launch_dir

        self._datasets: List[DatasetCache] = (self._define_datasets() if datasets_to_load == 'auto'
                                              else self._dataset_names_to_cache(datasets_to_load))

        self.candidate_pipelines = candidate_pipelines

        if self.candidate_pipelines is None:
            candidate_pipeline_paths = candidate_pipeline_paths or self._define_pipeline_paths()
            self._import_pipelines(candidate_pipeline_paths)

    def load(self, datasets: Union[List[str], Literal['auto']] = 'auto', n_best: int = 1) -> List[List[Model]]:
        if datasets != 'auto':
            datasets = self._dataset_names_to_cache(datasets)
            difference = set(d.name for d in datasets) - set(self.dataset_names)
            if difference:
                raise ValueError(f'Results for these datasets are not available: {difference}.')
        else:
            datasets = self._datasets

        models = []
        for dataset, candidate_pipelines in tqdm(list(zip(datasets, self.candidate_pipelines)),
                                                 desc='Selecting best models', unit='dataset'):
            best_performers = get_n_best_fedot_performers(dataset, candidate_pipelines, self.datasets_loader, n_best)
            models.append(best_performers)
        return models

    def _define_pipeline_paths(self) -> List[List[Path]]:
        if not self.launch_dir:
            raise ValueError('Launch dir or model paths must be provided!')

        dataset_names = self.dataset_names
        datasets_models_paths = dict(zip(dataset_names, [[]] * len(dataset_names)))

        for dataset_name in tqdm(dataset_names, desc='Defining model paths', unit='dataset'):
            for model_path in self.launch_dir.joinpath(dataset_name).glob(r'FEDOT*\*\*\launch_*.json'):
                datasets_models_paths[dataset_name].append(model_path)

        return [datasets_models_paths[dataset.name] for dataset in self._datasets]

    def _import_pipelines(self, candidate_pipeline_paths: List[List[PathType]]):
        candidate_pipelines = []
        for dataset, paths in tqdm(list(zip(self._datasets, candidate_pipeline_paths)),
                                   desc='Importing pipelines', unit='dataset'):
            candidates_for_dataset = [Pipeline.from_serialized(str(p)) for p in paths]
            if not candidates_for_dataset:
                self.log.warning(f'No pipelines found for the dataset "{dataset.name}".')
            candidate_pipelines.append(candidates_for_dataset)
        self.candidate_pipelines = candidate_pipelines

    def _define_datasets(self) -> List[DatasetCache]:
        if not self.launch_dir:
            raise ValueError('Launch dir or datasets must be provided!')

        datasets = list({p.parents[2].name for p in self.launch_dir.glob(r'*\FEDOT*\*\launch_0')})
        datasets.sort()
        datasets = self._dataset_names_to_cache(datasets)
        return datasets

    @property
    def dataset_names(self):
        return [d.name if isinstance(d, DatasetCache) else d for d in self._datasets]

    @staticmethod
    def _dataset_names_to_cache(datasets: List[Union[str, DatasetCache]]) -> List[DatasetCache]:
        new_list = []
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset = DatasetCache(dataset)
            new_list.append(dataset)
        return new_list
