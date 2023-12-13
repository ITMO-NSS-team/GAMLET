from copy import copy
from typing import Any, Dict, Sequence

from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self

from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.evaluated_model import EvaluatedModel


class DatasetModelsFitnessScaler:
    def __init__(self, scaler_class=MinMaxScaler):
        self.scaler_class = scaler_class
        self.scalers: Dict[DatasetIDType, Any] = {}

    def fit(self, dataset_ids: Sequence[DatasetIDType], models: Sequence[Sequence[EvaluatedModel]]) -> Self:
        for dataset_id, dataset_models in zip(dataset_ids, models):
            scaler = self.scaler_class()
            self.scalers[dataset_id] = scaler
            fitness_values_array = [model.fitness.values for model in dataset_models]
            scaler.fit(fitness_values_array)
        return self

    def transform(self, dataset_ids: Sequence[DatasetIDType], models: Sequence[Sequence[EvaluatedModel]]):
        new_models = [[copy(model) for model in dataset_models] for dataset_models in models]
        for dataset_id, dataset_models in zip(dataset_ids, new_models):
            scaler = self.scalers[dataset_id]
            fitness_values_array = [model.fitness.values for model in dataset_models]
            fitness_values_array = scaler.transform(fitness_values_array)
            for model, fitness_values in zip(dataset_models, fitness_values_array):
                fitness = copy(model.fitness)
                fitness.values = fitness_values
                model.fitness = fitness
        return new_models

    def fit_transform(self,
                      dataset_ids: Sequence[DatasetIDType],
                      models: Sequence[Sequence[EvaluatedModel]]) -> Sequence[Sequence[EvaluatedModel]]:
        self.fit(dataset_ids, models)
        return self.transform(dataset_ids, models)
