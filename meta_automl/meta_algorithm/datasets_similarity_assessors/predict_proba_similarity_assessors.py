from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from meta_automl.meta_algorithm.datasets_similarity_assessors.datasets_similarity_assessor import \
    DatasetsSimilarityAssessor


class PredictProbaSimilarityAssessor(DatasetsSimilarityAssessor):
    def __init__(self, model, n_best: int = 1):
        self._model = model
        self._model_encoding_map: Dict[str, Pipeline] = {}
        self.n_best = n_best

    @staticmethod
    def _preprocess_meta_features(meta_features: pd.DataFrame) -> pd.DataFrame:
        return meta_features.dropna(axis=1, how='any')

    def fit(self, meta_features: pd.DataFrame, models: List[Pipeline]):
        meta_features = self._preprocess_meta_features(meta_features)
        y = []
        for pipeline in models:
            descriptive_id = pipeline.descriptive_id
            self._model_encoding_map[descriptive_id] = pipeline
            y.append(descriptive_id)
        self._model.fit(meta_features, y)

    def get_similar(self, meta_features: pd.DataFrame) -> List[List[Pipeline]]:
        meta_features = self._preprocess_meta_features(meta_features)
        prediction = self._model.predict_proba(meta_features)
        decoded_prediction = []
        for pred in prediction:
            pred = list(pred)
            models = []
            for _ in range(self.n_best):
                class_idx = np.argmax(pred)
                model_encoding = self._model.classes_[class_idx]
                model = self._model_encoding_map[model_encoding]
                models.append(model)
                pred.pop(class_idx)
            decoded_prediction.append(models)

        return decoded_prediction


class KNNSimilarityAssessor(PredictProbaSimilarityAssessor):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, n_best: int = 1):
        model_params = model_params or dict()
        model = KNeighborsClassifier(**model_params)
        super().__init__(model, n_best)
