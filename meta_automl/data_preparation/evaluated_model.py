from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Union

from golem.core.optimisers.fitness import Fitness

from meta_automl.data_preparation.dataset import DatasetBase

PredictorType = Any


@dataclass
class EvaluatedModel:
    predictor: PredictorType
    fitness: Fitness
    fitness_metric_name: Union[str, Sequence[str]]
    dataset: DatasetBase
    metadata: Dict[str, Any] = field(default_factory=dict)
