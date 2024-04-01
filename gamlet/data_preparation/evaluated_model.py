from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Union

from golem.core.optimisers.fitness import Fitness

from gamlet.data_preparation.dataset import DatasetBase

PredictorType = Any


@dataclass
class EvaluatedModel:
    predictor: PredictorType
    metrics: Dict[str, Fitness]
    dataset: DatasetBase
    metadata: Dict[str, Any] = field(default_factory=dict)
