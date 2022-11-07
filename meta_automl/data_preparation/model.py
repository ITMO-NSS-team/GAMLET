from dataclasses import dataclass
from typing import Any

from fedot.core.optimisers.fitness import Fitness

from meta_automl.data_preparation.dataset import DatasetCache


@dataclass
class Model:
    predictor: Any
    fitness: Fitness
    data: DatasetCache
