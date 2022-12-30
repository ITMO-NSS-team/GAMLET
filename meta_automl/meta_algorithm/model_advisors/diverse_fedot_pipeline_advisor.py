from typing import Callable, List, Iterable

from fedot.core.dag.linked_graph import get_distance_between
from fedot.core.pipelines.pipeline import Pipeline

from meta_automl.data_preparation.model import Model
from meta_automl.meta_algorithm.datasets_similarity_assessors import DatasetsSimilarityAssessor
from meta_automl.meta_algorithm.model_advisors import SimpleSimilarityModelAdvisor


class DiverseFEDOTPipelineAdvisor(SimpleSimilarityModelAdvisor):
    def __init__(self,
                 fitted_similarity_assessor: DatasetsSimilarityAssessor,
                 minimal_distance: int = 1,
                 distance_func: Callable[[Pipeline, Pipeline], int] = get_distance_between):
        super().__init__(fitted_similarity_assessor)
        self.minimal_distance = minimal_distance
        self.distance_func = distance_func

    def _predict_single(self, similar_dataset_names: Iterable[str]) -> List[Model]:
        dataset_advice = super()._predict_single(similar_dataset_names)
        first_model = dataset_advice[0]
        diverse_dataset_advice = [first_model]
        for model in dataset_advice[1:]:
            if self.distance_func(first_model.predictor, model.predictor) > self.minimal_distance:
                diverse_dataset_advice.append(model)
        return diverse_dataset_advice
