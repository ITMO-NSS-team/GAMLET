from abc import ABC
from typing import Optional, Dict, Any, List, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from meta_automl.meta_algorithm.datasets_similarity_assessors.datasets_similarity_assessor import \
    DatasetsSimilarityAssessor


class ModelBasedSimilarityAssessor(ABC, DatasetsSimilarityAssessor):
    def __init__(self, model, n_best: int = 1):
        self._inner_model = model
        self.n_best = n_best
        self._datasets: Optional[Iterable[str]] = None


class KNNSimilarityAssessor(ModelBasedSimilarityAssessor):
    def __init__(self, n_neighbors: int = 1, **model_params):
        model = NearestNeighbors(n_neighbors=n_neighbors, **model_params)
        super().__init__(model, n_neighbors)

    def fit(self, meta_features: pd.DataFrame, datasets: Iterable[str]):
        meta_features = self.preprocess_meta_features(meta_features)
        self._datasets = np.array(datasets)
        self._inner_model.fit(meta_features)

    @staticmethod
    def preprocess_meta_features(meta_features: pd.DataFrame) -> pd.DataFrame:
        return meta_features.dropna(axis=1, how='any')

    def predict(self, meta_features: pd.DataFrame, return_distance: bool = False) -> Iterable[Iterable[str]]:
        dataset_indexes = self._inner_model.kneighbors(meta_features, return_distance=return_distance)
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
