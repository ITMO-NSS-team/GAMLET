from typing import Callable, List, Iterable, Optional

from fedot.core.pipelines.pipeline import Pipeline
from golem.core.dag.linked_graph import get_distance_between

from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import DatasetsSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import SimpleSimilarityModelAdvisor


class DiverseFEDOTPipelineAdvisor(SimpleSimilarityModelAdvisor):
    def __init__(self,
                 fitted_similarity_assessor: DatasetsSimilarityAssessor,
                 n_best_to_advise: Optional[int] = None,
                 minimal_distance: float = 1,
                 distance_func: Callable[[Pipeline, Pipeline], float] = get_distance_between):
        super().__init__(fitted_similarity_assessor)
        self.minimal_distance = minimal_distance
        self.n_best_to_advise = n_best_to_advise
        self.distance_func = distance_func

    def _predict_single(self, similar_dataset_names: Iterable[str]) -> List[Model]:
        dataset_advice = super()._predict_single(similar_dataset_names)
        first_model = dataset_advice[0]
        diverse_dataset_advice = [first_model]
        for model in dataset_advice[1:]:
            if self.distance_func(first_model.predictor, model.predictor) > self.minimal_distance:
                diverse_dataset_advice.append(model)

        if self.n_best_to_advise is not None:
            diverse_dataset_advice = list(sorted(diverse_dataset_advice, key=lambda m: m.fitness, reverse=True))
            diverse_dataset_advice = diverse_dataset_advice[:self.n_best_to_advise]
        return diverse_dataset_advice
