from abc import abstractmethod
from typing import Dict, Iterable, List, Self, Sequence

import pandas as pd

from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import DatasetsSimilarityAssessor


class ModelAdvisor:
    """Root class for model recommendation.

    Suggests pre-saved models for for the most similar datasets.
    """

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[List[Model]]:
        raise NotImplementedError()


class SimpleSimilarityModelAdvisor(ModelAdvisor):
    """Dataset similarity-based model advisor.

    Recommends stored models that are correlated with similar datasets.

    Example:
        >>> ???
    """

    def __init__(self, fitted_similarity_assessor: DatasetsSimilarityAssessor) -> None:
        """
        Args:
            fitted_similarity_assessor: dataset similarity assessor.
        """
        self.similarity_assessor = fitted_similarity_assessor
        self.best_models: Dict[DatasetIDType, Sequence[Model]] = {}

    @property
    def datasets(self) -> List[str]:
        """
        Get the names of the datasets.

        Returns:
            List of dataset names.
        """
        return self.similarity_assessor.datasets

    def fit(self, dataset_names_to_best_pipelines: Dict[DatasetIDType, Sequence[Model]]) -> Self:
        """Update the collection of recommended pipelines.

        Args:
            dataset_names_to_best_pipelines: Dictionary of mapped dataset names to a collection of models.
        """
        self.best_models.update(dataset_names_to_best_pipelines)
        return self

    def predict(self, meta_features: pd.DataFrame) -> List[List[Model]]:
        """Advises pipelines based on meta-learning.

        Args:
            meta_features: Pandas dataframe of meta features.

        Returns:
            List of lists of advised pipelines.
        """
        assessor_predictions = self.similarity_assessor.predict(meta_features)
        advised_pipelines = []
        for similar_datasets in assessor_predictions:
            advised_pipelines.append(self._predict_single(similar_datasets))
        return advised_pipelines

    def _predict_single(self, similar_dataset_ids: Iterable[DatasetIDType]) -> List[Model]:
        """Advises pipelines based on identifiers of similar datasets.

        Args:
            similar_dataset_ids: Iterable object of dataset ids.

        Returns:
            List of recommended models.
        """
        dataset_pipelines = []
        for dataset_id in similar_dataset_ids:
            dataset_pipelines += list(self.best_models.get(dataset_id))
        return dataset_pipelines
