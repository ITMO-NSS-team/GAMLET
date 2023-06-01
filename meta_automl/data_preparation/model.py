from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from golem.core.optimisers.fitness import Fitness

from meta_automl.data_preparation.dataset import DatasetBase


@dataclass
class Model:
    predictor: Any
    fitness: Fitness
    fitness_metric_name: str
    dataset: DatasetBase
    metadata: Dict[str, Any] = field(default_factory=dict)
