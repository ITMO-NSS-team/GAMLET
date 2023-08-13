from abc import ABC
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.meta_algorithm.datasets_similarity_assessors.datasets_similarity_assessor import (
    DatasetsSimilarityAssessor,
)


class ModelBasedSimilarityAssessor(ABC, DatasetsSimilarityAssessor):
    """
    Assesses dataset similarity based on Model meta-features.
    For a given dataset, provides list of similar datasets and optionally calculates similarity measures.
    """

    def __init__(self, model, n_best: int = 1):
        """
        Args:
            model:
                Model used to calculate neighbor searches.
            n_best: default=1
                Number of neighbors to use.
        """
        self._inner_model = model
        self.n_best = n_best
        self._datasets: Optional[Iterable[DatasetIDType]] = None


class KNeighborsBasedSimilarityAssessor(ModelBasedSimilarityAssessor):
    def __init__(self, n_neighbors: int = 1, **model_params):
        """
        Args:
            n_neighbors: Number of neighbors to use for queries.
            **model_params: Additional parameters passed to `NearestNeighbors` classifier.
        """
        model = NearestNeighbors(n_neighbors=n_neighbors, **model_params)
        super().__init__(model, n_neighbors)

    def fit(self, meta_features: pd.DataFrame, datasets: Iterable[DatasetIDType]) -> None:
        """Fit the meta features to the model.

        Args:
            meta_features: Pandas dataframe of model meta features.
            datasets: Iterable object of dataset names.
        """
        meta_features = self.preprocess_meta_features(meta_features)
        self._datasets = np.array(datasets)
        self._inner_model.fit(meta_features)

    @staticmethod
    def preprocess_meta_features(meta_features: pd.DataFrame) -> pd.DataFrame:
        """Remove missing values from meta features.

        Args:
            meta_features: Pandas dataframe of model meta features.
        Returns:
            Pandas dataframe cleared of missing values.
        """
        return meta_features.dropna(axis=1, how="any")

    def predict(self, meta_features: pd.DataFrame, return_distance: bool = False) -> Iterable[Iterable[DatasetIDType]]:
        dataset_indexes = self._inner_model.kneighbors(meta_features, return_distance=return_distance)
        """Predict the closest dataset names to the passed meta features.

        Args:
            meta_features:
                Pandas dataframe of model meta features.
            return_distance: default=False
                Whether or not to return the distances.
        Return:
            Returns iterable object of dataset names.
            Optionally when 'return_distance' == True returns measure of similarity to the neighbors of each point.
        """
        if return_distance:
            distances, dataset_indexes = dataset_indexes
            dataset_names = np.take(self._datasets, dataset_indexes, axis=0)
            return distances, dataset_names
        else:
            return np.take(self._datasets, dataset_indexes, axis=0)

    @property
    def datasets(self) -> Optional[Iterable[str]]:
        return self._datasets

    @property
    def feature_names(self) -> List[str]:
        return self._inner_model.feature_names_in_

    def _preprocess_predict_features(self, meta_features: pd.DataFrame) -> pd.DataFrame:
        return meta_features[self.feature_names]
