from typing import Optional, Dict, Any, List, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from meta_automl.meta_algorithm.datasets_similarity_assessors.datasets_similarity_assessor import \
    DatasetsSimilarityAssessor


class PredictProbaSimilarityAssessor(DatasetsSimilarityAssessor):
    def __init__(self, model, n_best: int = 1):
        self._inner_model = model
        self.n_best = n_best

    @property
    def datasets(self) -> List[str]:
        return self._inner_model.classes_

    @property
    def feature_names(self) -> List[str]:
        return self._inner_model.feature_names_in_

    @staticmethod
    def preprocess_meta_features(meta_features: pd.DataFrame) -> pd.DataFrame:
        return meta_features.dropna(axis=1, how='any')

    def _preprocess_predict_features(self, meta_features: pd.DataFrame) -> pd.DataFrame:
        return meta_features[self.feature_names]

    def fit(self, meta_features: pd.DataFrame, datasets: Iterable[str]):
        meta_features = self.preprocess_meta_features(meta_features)
        self._inner_model.fit(meta_features, datasets)

    def predict_proba(self, meta_features: pd.DataFrame) -> List[List[float]]:
        return self._inner_model.predict_proba(meta_features)

    def predict(self, meta_features: pd.DataFrame) -> List[List[str]]:
        meta_features = self._preprocess_predict_features(meta_features)
        predict_probs = self.predict_proba(meta_features)
        final_prediction = []
        for probabilities in predict_probs:
            probabilities = list(probabilities)
            predictions = []
            for _ in range(self.n_best):
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = self.datasets[predicted_class_idx]
                predictions.append(predicted_class)
                probabilities.pop(predicted_class_idx)
            final_prediction.append(predictions)

        return final_prediction


class KNNSimilarityAssessor(PredictProbaSimilarityAssessor):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, n_best: int = 1):
        model_params = model_params or dict()
        model = KNeighborsClassifier(**model_params)
        super().__init__(model, n_best)
