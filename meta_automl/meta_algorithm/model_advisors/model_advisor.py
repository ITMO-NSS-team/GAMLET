from abc import abstractmethod
from typing import Dict, Iterable, List, Self, Sequence

import pandas as pd

from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import DatasetsSimilarityAssessor


class ModelAdvisor:
    """Abstract class of Model Advisor."""

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[List[Model]]:
        raise NotImplementedError()


class SimpleSimilarityModelAdvisor(ModelAdvisor):
    """Simple similarity model advisor.

    Combines results from Models Loader and Datasets Similarity Assessor.
    Provides recommendations for models based on loaded data and similar datasets.
    Possible implementations allow for heuristic-based suggestions.

    Example:
        >>> ???
    """

    def __init__(self, fitted_similarity_assessor: DatasetsSimilarityAssessor) -> None:
        """
        Args:
            fitted_similarity_assessor (DatasetsSimilarityAssessor): abstract dataset similarity assessor.
        """
        self.similarity_assessor = fitted_similarity_assessor
        self.best_models: Dict[DatasetIDType, Sequence[Model]] = {}

    @property
    def datasets(self) -> List[str]:
        """
        Access names of dataset names.

        Returns:
            List: list of datasets names.
        """
        return self.similarity_assessor.datasets

    def fit(self, dataset_names_to_best_pipelines: Dict[DatasetIDType, Sequence[Model]]) -> Self:
        """Map each Dataset ID type to best dataset pipelines.

        Args:
            dataset_names_to_best_pipelines: Dataset names.
        """
        self.best_models.update(dataset_names_to_best_pipelines)
        return self

    def predict(self, meta_features: pd.DataFrame) -> List[List[Model]]:
        """Advises pipelines by meta features.

        Args:
            meta_features: pandas dataframe of meta features.

        Returns:
            List: List of list of advised pipelines.
        """
        assessor_predictions = self.similarity_assessor.predict(meta_features)
        advised_pipelines = []
        for similar_datasets in assessor_predictions:
            advised_pipelines.append(self._predict_single(similar_datasets))
        return advised_pipelines

    def _predict_single(self, similar_dataset_ids: Iterable[DatasetIDType]) -> List[Model]:
        """Advises pipelines based on similarity of dataset ids.

        Args:
            similar_dataset_ids: Iterable object of dataset ids types.

        Returns:
            List: List of dataset model pipelines.
        """
        dataset_pipelines = []
        for dataset_id in similar_dataset_ids:
            dataset_pipelines += list(self.best_models.get(dataset_id))
        return dataset_pipelines
