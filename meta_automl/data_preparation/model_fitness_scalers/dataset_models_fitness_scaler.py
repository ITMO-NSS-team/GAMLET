from copy import copy
from typing import Dict, Sequence, Type, TypeVar

from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self

from meta_automl.data_preparation.dataset.dataset_base import DatasetType_co
from meta_automl.data_preparation.evaluated_model import EvaluatedModel

ScalerType = TypeVar('ScalerType', OneToOneFeatureMixin, TransformerMixin)


class DatasetModelsFitnessScaler:
    def __init__(self, scaler_class: Type[ScalerType] = MinMaxScaler):
        self.scaler_class = scaler_class
        self.scalers: Dict[str, ScalerType] = {}

    def fit(self, models: Sequence[Sequence[EvaluatedModel]], datasets: Sequence[DatasetType_co]) -> Self:
        dataset_representations = map(repr, datasets)
        for dataset_repr, dataset_models in zip(dataset_representations, models):
            scaler = self.scaler_class()
            self.scalers[dataset_repr] = scaler
            fitness_values_array = [model.fitness.values for model in dataset_models]
            scaler.fit(fitness_values_array)
        return self

    def transform(self, models: Sequence[Sequence[EvaluatedModel]], datasets: Sequence[DatasetType_co]):
        new_models = [[copy(model) for model in dataset_models] for dataset_models in models]
        dataset_representations = map(repr, datasets)
        for dataset_repr, dataset_models in zip(dataset_representations, new_models):
            scaler = self.scalers[dataset_repr]
            fitness_values_array = [model.fitness.values for model in dataset_models]
            fitness_values_array = scaler.transform(fitness_values_array)
            for model, fitness_values in zip(dataset_models, fitness_values_array):
                fitness = copy(model.fitness)
                fitness.values = fitness_values
                model.fitness = fitness
        return new_models

    def fit_transform(self,
                      models: Sequence[Sequence[EvaluatedModel]],
                      datasets: Sequence[DatasetType_co]) -> Sequence[Sequence[EvaluatedModel]]:
        self.fit(models, datasets)
        return self.transform(models, datasets)
