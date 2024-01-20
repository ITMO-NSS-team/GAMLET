import logging
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from fedot.core.data.cv_folds import cv_generator
from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from tqdm import tqdm
from typing_extensions import Literal

from gamlet.data_preparation.dataset import DatasetBase
from gamlet.components.datasets_loaders import DatasetsLoader, OpenMLDatasetsLoader
from gamlet.data_preparation.evaluated_model import EvaluatedModel
from gamlet.data_preparation.file_system import PathType
from gamlet.components.models_loaders import ModelsLoader

logger = logging.getLogger(__file__)


def evaluate_classification_fedot_pipeline(pipeline, input_data):
    cv_folds = partial(cv_generator, input_data, folds=5)
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(ClassificationMetricsEnum.ROCAUC), cv_folds,
                                               eval_n_jobs=-1)
    fitness = objective_eval(pipeline)
    return fitness


def get_n_best_fedot_performers(dataset: DatasetBase,
                                pipelines: List[Pipeline],
                                n_best: int = 1) -> List[EvaluatedModel]:
    data = dataset.get_data()
    X, y_test = data.x.to_numpy(), data.y.to_numpy()
    input_data = InputData(idx=np.arange(0, len(X)), features=X, target=y_test, data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.classification))
    fitnesses = []
    models = []
    metric_name = 'roc_auc'
    for pipeline in tqdm(pipelines, desc='Evaluating pipelines'):
        fitness = evaluate_classification_fedot_pipeline(pipeline, input_data)
        fitnesses.append(fitness)
        models.append(EvaluatedModel(pipeline, fitness, metric_name, dataset))

    best_models = [models.pop(np.argmax(fitnesses)) for _ in range(min(n_best, len(pipelines)))]
    return best_models


class FEDOTPipelinesLoader(ModelsLoader):
    def __init__(self, datasets_to_load: Union[List[Union[DatasetBase, str]], Literal['auto']] = 'auto',
                 candidate_pipelines: Optional[List[List[Pipeline]]] = None,
                 candidate_pipeline_paths: Optional[List[List[PathType]]] = None,
                 launch_dir: Optional[PathType] = None,
                 datasets_loader: Optional[DatasetsLoader] = None):

        self.datasets_loader = datasets_loader or OpenMLDatasetsLoader(allow_names=True)

        self.launch_dir: Path = Path(launch_dir) if isinstance(launch_dir, str) else launch_dir

        self._datasets: List[DatasetBase] = (self._define_datasets() if datasets_to_load == 'auto'
                                             else self._get_datasets_from_names(datasets_to_load))

        self.candidate_pipelines = candidate_pipelines

        if self.candidate_pipelines is None:
            candidate_pipeline_paths = candidate_pipeline_paths or self._define_pipeline_paths()
            self._import_pipelines(candidate_pipeline_paths)

    def load(self, datasets: Union[List[str], Literal['auto']] = 'auto', n_best: int = 1) -> List[List[EvaluatedModel]]:
        if datasets != 'auto':
            datasets = self._get_datasets_from_names(datasets)
            difference = set(d.name for d in datasets) - set(self.dataset_ids)
            if difference:
                raise ValueError(f'Results for these datasets are not available: {difference}.')
        else:
            datasets = self._datasets

        models = []
        for dataset, candidate_pipelines in tqdm(list(zip(datasets, self.candidate_pipelines)),
                                                 desc='Selecting best models', unit='dataset'):
            best_performers = get_n_best_fedot_performers(dataset, candidate_pipelines, n_best)
            models.append(best_performers)
        return models

    def _define_pipeline_paths(self) -> List[List[Path]]:
        if not self.launch_dir:
            raise ValueError('Launch dir or model paths must be provided!')

        dataset_ids = self.dataset_ids
        datasets_models_paths = dict(zip(dataset_ids, [[]] * len(dataset_ids)))

        for dataset_name in tqdm(dataset_ids, desc='Defining model paths', unit='dataset'):
            for model_path in self.launch_dir.joinpath(dataset_name).glob(r'FEDOT*\*\*\launch_*.json'):
                datasets_models_paths[dataset_name].append(model_path)

        return [datasets_models_paths[dataset.name] for dataset in self._datasets]

    def _import_pipelines(self, candidate_pipeline_paths: List[List[PathType]]):
        candidate_pipelines = []
        for dataset, paths in tqdm(list(zip(self._datasets, candidate_pipeline_paths)),
                                   desc='Importing pipelines', unit='dataset'):
            candidates_for_dataset = [Pipeline.from_serialized(str(p)) for p in paths]
            if not candidates_for_dataset:
                logger.warning(f'No pipelines found for the dataset "{dataset}".')
            candidate_pipelines.append(candidates_for_dataset)
        self.candidate_pipelines = candidate_pipelines

    def _define_datasets(self) -> List[DatasetBase]:
        if not self.launch_dir:
            raise ValueError('Launch dir or datasets must be provided!')

        datasets = sorted({p.parents[2].name for p in self.launch_dir.glob(r'*\FEDOT*\*\launch_0')})
        datasets = self._get_datasets_from_names(datasets)
        return datasets

    @property
    def dataset_ids(self):
        return [d.name if isinstance(d, DatasetBase) else d for d in self._datasets]

    def _get_datasets_from_names(self, datasets: List[Union[str, DatasetBase]]) -> List[DatasetBase]:
        new_list = []
        for dataset in datasets:
            if not isinstance(dataset, DatasetBase):
                dataset = self.datasets_loader.load_single(dataset)
            new_list.append(dataset)
        return new_list
