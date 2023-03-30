from abc import abstractmethod
from typing import List, Dict, Iterable, Optional

import pandas as pd

from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import DatasetsSimilarityAssessor


class ModelAdvisor:

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[List[Model]]:
        raise NotImplementedError()


class SimpleSimilarityModelAdvisor(ModelAdvisor):
    def __init__(self, fitted_similarity_assessor: DatasetsSimilarityAssessor):
        self.similarity_assessor = fitted_similarity_assessor
        self.best_models: Dict[str, List[Model]] = {}

    @property
    def datasets(self):
        return self.similarity_assessor.datasets

    def fit(self, dataset_names_to_best_pipelines: Dict[str, List[Model]]):
        self.best_models.update(dataset_names_to_best_pipelines)
        return self

    def predict(self, meta_features: pd.DataFrame) -> List[List[Model]]:
        assessor_predictions = self.similarity_assessor.predict(meta_features)
        advised_pipelines = []
        for similar_datasets in assessor_predictions:
            advised_pipelines.append(self._predict_single(similar_datasets))
        return advised_pipelines

    def _predict_single(self, similar_dataset_names: Iterable[str]) -> List[Model]:
        dataset_pipelines = []
        for dataset_name in similar_dataset_names:
            dataset_pipelines += self.best_models.get(dataset_name)
        return dataset_pipelines
