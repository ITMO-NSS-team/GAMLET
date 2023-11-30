from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.evaluated_model import EvaluatedModel


class ModelAdvisor(ABC):
    """Root class for model recommendation.

    Suggests pre-saved models for the most similar datasets.
    """

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[List[EvaluatedModel]]:
        raise NotImplementedError()


class DatasetSimilarityModelAdvisor(ModelAdvisor):
    """Dataset similarity-based model advisor.

    Recommends stored models that are correlated with similar datasets.
    """

    def __init__(self, n_best_to_advise: Optional[int] = None):
        """
        Args:
            n_best_to_advise: Number of models to output. Defaults to `None` => all pipelines from memory.

        """
        self.n_best_to_advise = n_best_to_advise
        self.best_models: Dict[DatasetIDType, Sequence[EvaluatedModel]] = {}

    def fit(self, dataset_ids: Sequence[DatasetIDType], models: Sequence[Sequence[EvaluatedModel]]):
        """Update the collection of recommended pipelines.

        Args:
            dataset_ids: sequence of dataset ids.
            models: sequence of model sequences for the datasets.
        Returns:
            Self instance.
        """
        self.best_models.update(dict(zip(dataset_ids, models)))
        return self

    def predict(self, dataset_ids: Sequence[Sequence[DatasetIDType]]) -> List[List[EvaluatedModel]]:
        """Advises pipelines based on meta-learning.

        Args:
            dataset_ids: sequence of sequences, storing dataset ids, for which the best models should be retrieved.

        Returns:
            List of lists of advised pipelines.
        """
        advised_pipelines = []
        for similar_datasets in dataset_ids:
            advised_pipelines.append(self._predict_single(similar_datasets))
        return advised_pipelines

    def _predict_single(self, dataset_ids: Sequence[DatasetIDType],
                        n_best_to_advise: Optional[int] = None) -> List[EvaluatedModel]:
        """ Advises pipelines based on identifiers of datasets,
            looking for similar datasets and corresponding models in its knowledge base2.

        Args:
            dataset_ids: Iterable object of dataset ids.
            n_best_to_advise: default=None
                Number of models to output. Defaults to `None` => all pipelines from memory.

        Returns:
            List of recommended models.
        """
        n_best_to_advise = n_best_to_advise or self.n_best_to_advise
        dataset_models = self._get_all_models_for_datasets(dataset_ids)

        dataset_models = self._sort_models_by_fitness(dataset_models, n_best_to_advise)

        return dataset_models

    def _get_all_models_for_datasets(self, similar_dataset_ids: Sequence[DatasetIDType]) -> List[EvaluatedModel]:
        dataset_models = []
        for dataset_id in similar_dataset_ids:
            dataset_models += list(self.best_models.get(dataset_id))
        return dataset_models

    @staticmethod
    def _sort_models_by_fitness(models: Sequence[EvaluatedModel],
                                n_best_to_advise: Union[int, None]) -> List[EvaluatedModel]:
        if n_best_to_advise is not None:
            models = list(sorted(models, key=lambda m: m.fitness, reverse=True))
            models = models[: n_best_to_advise]
        return models
