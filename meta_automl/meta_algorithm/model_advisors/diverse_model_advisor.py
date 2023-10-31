from typing import Callable, Iterable, List, Optional

from fedot.core.pipelines.pipeline import Pipeline
from golem.core.dag.linked_graph import get_distance_between

from meta_automl.data_preparation.dataset import DatasetIDType
from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.model_advisors import DatasetSimilarityModelAdvisor


class DiverseModelAdvisor(DatasetSimilarityModelAdvisor):
    """Diverse model advisor.

    Provides diverse recommendations for models based on loaded data and datasets.
    """

    def __init__(
            self,
            n_best_to_advise: Optional[int] = None,
            minimal_distance: float = 1.,
            distance_func: Callable[[Pipeline, Pipeline], float] = get_distance_between,
    ) -> None:
        """
        Args:
            n_best_to_advise: default=None
                Number of models to output. Defaults to `None` => all pipelines from memory.
            minimal_distance: default=1.0
                Minimal distance between first model and others.
            distance_func: default=`get_distance_between`
                Function that calculates distance from `pipeline_1` to `pipeline_2`.
        """
        super().__init__()
        self.minimal_distance = minimal_distance
        self.n_best_to_advise = n_best_to_advise
        self.distance_func = distance_func

    def _predict_single(self, similar_dataset_ids: Iterable[DatasetIDType], n_best_to_advise: Optional[int] = None,
                        minimal_distance: Optional[float] = None) -> List[Model]:
        """Advices list of dataset names closer to the most similar dataset.

        Args:
            similar_dataset_ids: Iterable object of dataset names.
            n_best_to_advise: default=None
                Number of models to output. Defaults to `None` => all pipelines from memory.

        Returns:
            List of advised models.
        """
        n_best_to_advise = n_best_to_advise or self.n_best_to_advise
        minimal_distance = minimal_distance or self.minimal_distance

        dataset_models = self._get_all_models_for_datasets(similar_dataset_ids)

        first_model = dataset_models[0]
        diverse_dataset_advice = [first_model]
        for model in dataset_models[1:]:
            if self.distance_func(first_model.predictor, model.predictor) >= minimal_distance:
                diverse_dataset_advice.append(model)

        diverse_dataset_advice = self._sort_models_by_fitness(diverse_dataset_advice, n_best_to_advise)

        return diverse_dataset_advice
